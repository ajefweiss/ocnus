use crate::{
    base::{OcnusModel, OcnusModelError, ScObs, ScObsConf},
    coords::{CCGeometry, ECGeometry, OcnusCoords, XCState, param_value},
    math::bessel_jn,
    models::concat_strs,
    obser::{MeasureInSituMagneticFields, ObserVec},
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
) -> Result<(T, T), OcnusModelError<T>>
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
    let y_offset =
        param_value("y", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
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

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

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
                let b_linearized = b / (T::one() - y_offset.powi(2));

                // Bessel function evaluation uses 11 terms.
                let chi_lff = b_linearized
                    * radius_linearized
                    * sign
                    * bessel_jn(alpha * mu * radius_linearized, 1);

                let xi_lff: T = T::two_pi()
                    * mu
                    * radius_linearized.powi(2)
                    * b_linearized
                    * bessel_jn(alpha * mu * radius_linearized, 0);

                Ok((chi_lff, xi_lff))
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
) -> Result<(T, T), OcnusModelError<T>>
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
    let y_offset =
        param_value("y", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
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

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    let (mu, _nu, _z) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                let b_linearized = b / (T::one() - y_offset.powi(2));

                let chi_ut = mu * radius_linearized.powi(2) * b_linearized * tau
                    / (T::one() + (tau * mu * radius_linearized).powi(2));
                let xi_ut = T::two_pi() * mu * radius_linearized.powi(2) * b_linearized
                    / (T::one() + (tau * mu * radius_linearized).powi(2));

                Ok((chi_ut, xi_ut))
            }
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

/// Magnetic field configuration as is used in Nieves-Chinchilla et al. (2018).
pub fn ec_hybrid_obs<T, const D: usize>(
    q: Vector3<T>,
    names: &SVector<&'static str, D>,
    params: &SVector<T, D>,
) -> Result<(T, T), OcnusModelError<T>>
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let y_offset =
        param_value("y", names, &params.as_view::<Const<D>, U1, U1, Const<D>>()).unwrap();
    let radius = param_value(
        "radius",
        names,
        &params.as_view::<Const<D>, U1, U1, Const<D>>(),
    )
    .unwrap();
    let b = param_value(
        "b_scale",
        names,
        &params.as_view::<Const<D>, U1, U1, Const<D>>(),
    )
    .unwrap();
    let lambda = param_value(
        "lambda",
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
    let tau = param_value(
        "tau",
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
                let b_linearized = b / (T::one() - y_offset.powi(2));
                let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

                // Bessel function evaluation uses 11 terms.
                let chi_lff = b_linearized
                    * radius_linearized
                    * sign
                    * bessel_jn(alpha * mu * radius_linearized, 1);
                let xi_lff: T = T::two_pi()
                    * mu
                    * radius_linearized.powi(2)
                    * b_linearized
                    * bessel_jn(alpha * mu * radius_linearized, 0);

                // UT terms.
                let chi_ut = mu * radius_linearized.powi(2) * b_linearized * tau
                    / (T::one() + (tau * mu * radius_linearized).powi(2));
                let xi_ut = T::two_pi() * mu * radius_linearized.powi(2) * b_linearized
                    / (T::one() + (tau * mu * radius_linearized).powi(2));

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
            Self: Send + Sync,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`].")]
            pub fn new(pdf: P) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        impl<T, P>
            MeasureInSituMagneticFields<
                T,
                { $coords::<f32>::PARAMS_COUNT + $params.len() },
                (),
                XCState<T>,
            > for $model<T, P>
        where
            T: Copy + RealField + Sum,
            Self: Send + Sync,
        {
            fn observe_mag3(
                &self,
                scobs: &ScObs<T>,
                params: &SVector<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                _fm_state: &(),
                cs_state: &XCState<T>,
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

        // Re-implement the OcnusCoords trait because we have no inheritance.
        // Here we make use of the fact that the parameters for the coordinates are at the front
        // and we pass on smaller fixed views of each parameter vector.
        impl<T, P> OcnusCoords<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }, XCState<T>>
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
                cs_state: &XCState<T>,
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
                cs_state: &XCState<T>,
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
                cs_state: &mut XCState<T>,
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
                cs_state: &XCState<T>,
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
                cs_state: &XCState<T>,
            ) -> Option<Vector3<T>> {
                $coords::transform_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }
        }

        impl<T, P> OcnusModel<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }, (), XCState<T>>
            for $model<T, P>
        where
            T: Copy + Default + RealField + SampleUniform,
            P: for<'x> Deserialize<'x> + Serialize,
            for<'x> &'x P: Density<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
            StandardNormal: Distribution<T>,
            usize: AsPrimitive<T>,
            Self: OcnusCoords<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }, XCState<T>>,
        {
            const RCS: usize = 128;

            fn forward(
                &self,
                time_step: T,
                params: &VectorView<T, Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>>,
                _fm_state: &mut (),
                cs_state: &mut XCState<T>,
            ) -> Result<(), OcnusModelError<T>> {
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
                _fm_state: &mut (),
                cs_state: &mut XCState<T>,
            ) -> Result<(), OcnusModelError<T>> {
                Self::initialize_cs(params, cs_state);

                Ok(())
            }

            fn observe_ics_basis(
                &self,
                scobs: &ScObs<T>,
                params: &VectorView<T, Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>>,
                _fm_state: &(),
                cs_state: &XCState<T>,
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

impl_cylm!(
    CCLFFModel,
    CCGeometry,
    "Circular-cylindrical linear force-free magnetic flux rope model.",
    ["velocity", "b_scale", "alpha"],
    cc_lff_chi_xi
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
    "Elliptic-cylindrical uniform twist magnetic flux rope model.",
    ["velocity", "b_scale", "lambda", "alpha", "tau"],
    ec_hybrid_obs
);

#[cfg(test)]
mod tests {
    use std::f32;

    use super::*;
    use crate::{
        base::{OcnusEnsbl, ScObsSeries},
        obser::NullNoise,
        stats::{ConstantDensity, MultivariateDensity, UniformDensity},
    };
    use approx::ulps_eq;
    use nalgebra::{DMatrix, SVector};

    #[test]
    fn test_cclff_model() {
        let prior = MultivariateDensity::<_, 8>::new(&[
            UniformDensity::new((-1.0, 1.0)).unwrap(),
            UniformDensity::new((0.5, 1.0)).unwrap(),
            UniformDensity::new((0.05, 0.1)).unwrap(),
            UniformDensity::new((0.1, 0.5)).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new((5.0, 100.0)).unwrap(),
            UniformDensity::new((-2.4, 2.4)).unwrap(),
            UniformDensity::new((0.0, 1.0)).unwrap(),
        ]);

        let range = (&prior).get_range();
        let model = CCLFFModel::new(prior);

        let sc = ScObsSeries::<f32>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f32 * 3600.0 * 2.0,
                ScObsConf::Position([1.0, 0.0, 0.0]),
            )
        }));

        let mut ensbl = OcnusEnsbl::new(1, range);
        let mut output = DMatrix::<ObserVec<f32, 3>>::zeros(sc.len(), 1);

        ensbl.ptpdf.particles_mut().set_column(
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
            ]),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &sc,
                &mut ensbl,
                &CCLFFModel::<f32, MultivariateDensity<f32, 8>>::observe_mag3,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            output[(0, 0)][1],
            19.56287,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(2, 0)][1],
            20.085495,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(4, 0)][2],
            -1.714362,
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
            UniformDensity::new((-1.0, 1.0)).unwrap(),
            UniformDensity::new((0.5, 1.0)).unwrap(),
            UniformDensity::new((0.05, 0.1)).unwrap(),
            UniformDensity::new((0.1, 0.5)).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new((5.0, 100.0)).unwrap(),
            UniformDensity::new((-2.4, 2.4)).unwrap(),
            UniformDensity::new((0.0, 1.0)).unwrap(),
        ]);

        let range = (&prior).get_range();
        let model = CCLFFModel::new(prior);

        let sc = ScObsSeries::<f32>::from_iterator(
            (0..2).map(|i| ScObs::new(0.0, ScObsConf::Position([i as f32, 0.0, 0.0]))),
        );

        let mut ensbl = OcnusEnsbl::new(1, range);
        let mut output = DMatrix::<ObserVec<f32, 3>>::zeros(sc.len(), 1);

        ensbl.ptpdf.particles_mut().set_column(
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
            ]),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &sc,
                &mut ensbl,
                &CCLFFModel::<f32, MultivariateDensity<f32, 8>>::observe_mag3,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(output[(0, 0)][1], 20.0));
        assert!(ulps_eq!(output[(0, 0)][2], 0.0));

        assert!(ulps_eq!(
            output[(1, 0)][1],
            0.0,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(1, 0)][2] / bessel_jn(2.4048254f32, 1),
            20.0,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
    }

    #[test]
    fn test_ccut_model() {
        let prior = MultivariateDensity::<_, 8>::new(&[
            UniformDensity::new((-1.0, 1.0)).unwrap(),
            UniformDensity::new((0.5, 1.0)).unwrap(),
            UniformDensity::new((0.05, 0.1)).unwrap(),
            UniformDensity::new((0.1, 0.5)).unwrap(),
            UniformDensity::new((0.0, 1.0)).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new((5.0, 100.0)).unwrap(),
            UniformDensity::new((-10.0, 10.0)).unwrap(),
        ]);

        let range = (&prior).get_range();
        let model = CCUTModel::new(prior);

        let sc = ScObsSeries::<f32>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f32 * 3600.0 * 2.0,
                ScObsConf::Position([1.0, 0.0, 0.0]),
            )
        }));

        let mut ensbl = OcnusEnsbl::new(1, range);
        let mut output = DMatrix::<ObserVec<f32, 3>>::zeros(sc.len(), 1);

        ensbl.ptpdf.particles_mut().set_column(
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
            ]),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &sc,
                &mut ensbl,
                &CCUTModel::<f32, MultivariateDensity<f32, 8>>::observe_mag3,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            output[(0, 0)][1],
            17.744316,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(2, 0)][1],
            19.713773,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(4, 0)][2],
            -2.347745,
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
            UniformDensity::new((-1.0, 1.0)).unwrap(),
            UniformDensity::new((0.5, 1.0)).unwrap(),
            UniformDensity::new((0.05, 0.1)).unwrap(),
            UniformDensity::new((0.1, 0.5)).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new((5.0, 100.0)).unwrap(),
            UniformDensity::new((-2.4, 2.4)).unwrap(),
            UniformDensity::new((0.0, 1.0)).unwrap(),
        ]);

        let range = (&prior).get_range();
        let model = CCUTModel::new(prior);

        let sc = ScObsSeries::<f32>::from_iterator(
            (0..2).map(|i| ScObs::new(0.0, ScObsConf::Position([i as f32, 0.0, 0.0]))),
        );

        let mut ensbl = OcnusEnsbl::new(1, range);
        let mut output = DMatrix::<ObserVec<f32, 3>>::zeros(sc.len(), 1);

        ensbl.ptpdf.particles_mut().set_column(
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
            ]),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &sc,
                &mut ensbl,
                &CCUTModel::<f32, MultivariateDensity<f32, 8>>::observe_mag3,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            output[(0, 0)][1],
            20.0,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(0, 0)][2],
            0.0,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));

        assert!(ulps_eq!(output[(1, 0)][1], 10.0, max_ulps = 5,));
        assert!(ulps_eq!(
            output[(1, 0)][2],
            10.0,
            epsilon = 32.0 * f32::EPSILON
        ));
    }

    #[test]
    fn test_ech_model() {
        let prior = MultivariateDensity::<_, 12>::new(&[
            UniformDensity::new((-1.0, 1.0)).unwrap(),
            UniformDensity::new((-1.0, 1.0)).unwrap(),
            UniformDensity::new((-1.0, 1.0)).unwrap(),
            UniformDensity::new((0.05, 0.1)).unwrap(),
            UniformDensity::new((0.1, 1.0)).unwrap(),
            UniformDensity::new((0.1, 0.5)).unwrap(),
            UniformDensity::new((0.0, 1.0)).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new((5.0, 100.0)).unwrap(),
            UniformDensity::new((0.0, 1.0)).unwrap(),
            UniformDensity::new((-10.0, 10.0)).unwrap(),
            UniformDensity::new((-10.0, 10.0)).unwrap(),
        ]);

        let range = (&prior).get_range();
        let model = ECHModel::new(prior);

        let sc = ScObsSeries::<f32>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f32 * 3600.0 * 2.0,
                ScObsConf::Position([1.0, 0.0, 0.0]),
            )
        }));

        let mut ensbl = OcnusEnsbl::new(1, range);
        let mut output = DMatrix::<ObserVec<f32, 3>>::zeros(sc.len(), 1);

        // UT

        ensbl.ptpdf.particles_mut().set_column(
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
            ]),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &sc,
                &mut ensbl,
                &ECHModel::<f32, MultivariateDensity<f32, 12>>::observe_mag3,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            output[(0, 0)][1],
            17.744316,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(2, 0)][1],
            19.713773,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));

        assert!(ulps_eq!(
            output[(4, 0)][2],
            -2.3477452,
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

        ensbl.ptpdf.particles_mut().set_column(
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
            ]),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &sc,
                &mut ensbl,
                &ECHModel::<f32, MultivariateDensity<f32, 12>>::observe_mag3,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            output[(0, 0)][1],
            19.56287,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(2, 0)][1],
            20.085491,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(4, 0)][2],
            -1.7143621,
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
