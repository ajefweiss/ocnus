use crate::{
    base::{OcnusModel, OcnusModelError, ScObs, ScObsConf},
    coords::{CCGeometry, ECGeometry, OcnusCoords, XCState},
    math::bessel_jn,
    models::concat_strs,
    obser::ObserVec,
    stats::Density,
};
use nalgebra::{
    Const, DefaultAllocator, Dim, DimName, OVector, RealField, SVector, Vector3, VectorView,
    VectorView3, allocator::Allocator,
};
use num_traits::AsPrimitive;
use paste::paste;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use std::{cmp::Ordering, iter::Sum, marker::PhantomData};

/// Linear force-free magnetic field Chi (nu) and Xi (s) functions.
pub fn cc_lff_chi_xi<T, D, M>(
    q: Vector3<T>,
    params: &OVector<T, D>,
    cs_state: &XCState<T>,
) -> Result<ObserVec<T, 3>, OcnusModelError<T>>
where
    T: Copy + RealField,
    D: DimName,
    M: OcnusCoords<T, D, XCState<T>>,
    SVector<T, 3>: Sum,
    DefaultAllocator: Allocator<D>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("b_scale", &params.as_view()).unwrap();
    let y_offset = M::param_value("y", &params.as_view()).unwrap();
    let alpha_signed = M::param_value("alpha", &params.as_view()).unwrap();
    let radius = M::param_value("radius", &params.as_view()).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    let (alpha, sign) = match alpha_signed
        .partial_cmp(&T::zero())
        .expect("alpha value is NaN")
    {
        Ordering::Less => (-alpha_signed, T::neg(T::one())),
        _ => (alpha_signed, T::one()),
    };

    let (mu, nu, z) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(ObserVec::default()),
            _ => {
                let b_linearized = b / (T::one() - y_offset.powi(2));

                if mu == T::zero() {
                    let b_q = Vector3::new(T::zero(), T::zero(), b_linearized);
                    let b_s = M::contravariant_vector(
                        &q.as_view(),
                        &b_q.as_view(),
                        &params.as_view(),
                        cs_state,
                    )
                    .expect("failed to construct contravariant basis");

                    Ok(ObserVec::<T, 3>::from(b_s))
                } else {
                    // Bessel function evaluation uses 11 terms.
                    let chi_lff = b_linearized
                        * radius_linearized
                        * sign
                        * bessel_jn(alpha * mu * radius_linearized, 1)
                        / M::detg(
                            &Vector3::new(mu, nu, z).as_view(),
                            &params.as_view(),
                            cs_state,
                        )
                        .unwrap();
                    let xi_lff: T = T::two_pi()
                        * mu
                        * radius_linearized.powi(2)
                        * b_linearized
                        * bessel_jn(alpha * mu * radius_linearized, 0)
                        / M::detg(
                            &Vector3::new(mu, nu, z).as_view(),
                            &params.as_view(),
                            cs_state,
                        )
                        .unwrap();

                    let b_q = Vector3::new(T::zero(), chi_lff, xi_lff);
                    let b_s = M::contravariant_vector(
                        &q.as_view(),
                        &b_q.as_view(),
                        &params.as_view(),
                        cs_state,
                    )
                    .expect("failed to construct contravariant basis");

                    Ok(ObserVec::<T, 3>::from(b_s))
                }
            }
        },
        None => Ok(ObserVec::default()),
    }
}

/// Uniform twist magnetic field observable.
pub fn cc_ut_chi_xi<T, D, M>(
    q: Vector3<T>,
    params: &OVector<T, D>,
    cs_state: &XCState<T>,
) -> Result<ObserVec<T, 3>, OcnusModelError<T>>
where
    T: Copy + RealField,
    D: DimName,
    M: OcnusCoords<T, D, XCState<T>>,
    SVector<T, 3>: Sum,
    DefaultAllocator: Allocator<D>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("b_scale", &params.as_view()).unwrap();
    let y_offset = M::param_value("y", &params.as_view()).unwrap();
    let tau = M::param_value("tau", &params.as_view()).unwrap();
    let radius = M::param_value("radius", &params.as_view()).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    let (mu, nu, z) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(ObserVec::default()),
            _ => {
                let b_linearized = b / (T::one() - y_offset.powi(2));

                if mu == T::zero() {
                    let b_q = Vector3::new(T::zero(), T::zero(), b_linearized);
                    let b_s = M::contravariant_vector(
                        &q.as_view(),
                        &b_q.as_view(),
                        &params.as_view(),
                        cs_state,
                    )
                    .expect("failed to construct contravariant basis");

                    Ok(ObserVec::<T, 3>::from(b_s))
                } else {
                    let chi_ut = mu * radius_linearized.powi(2) * b_linearized * tau
                        / (T::one() + (tau * mu * radius_linearized).powi(2))
                        / M::detg(
                            &Vector3::new(mu, nu, z).as_view(),
                            &params.as_view(),
                            cs_state,
                        )
                        .unwrap();
                    let xi_ut = T::two_pi() * mu * radius_linearized.powi(2) * b_linearized
                        / (T::one() + (tau * mu * radius_linearized).powi(2))
                        / M::detg(
                            &Vector3::new(mu, nu, z).as_view(),
                            &params.as_view(),
                            cs_state,
                        )
                        .unwrap();

                    let b_q = Vector3::new(T::zero(), chi_ut, xi_ut);
                    let b_s = M::contravariant_vector(
                        &q.as_view(),
                        &b_q.as_view(),
                        &params.as_view(),
                        cs_state,
                    )
                    .expect("failed to construct contravariant basis");

                    Ok(ObserVec::<T, 3>::from(b_s))
                }
            }
        },
        None => Ok(ObserVec::default()),
    }
}

/// Magnetic field configuration as is used in Nieves-Chinchilla et al. (2018).
pub fn ec_hybrid_obs<T, D, M>(
    q: Vector3<T>,
    params: &OVector<T, D>,
    cs_state: &XCState<T>,
) -> Result<ObserVec<T, 3>, OcnusModelError<T>>
where
    T: Copy + RealField,
    D: DimName,
    M: OcnusCoords<T, D, XCState<T>>,
    SVector<T, 3>: Sum,
    DefaultAllocator: Allocator<D>,
{
    // Extract parameters using their identifiers.
    let y_offset = M::param_value("y", &params.as_view()).unwrap();
    let radius = M::param_value("radius", &params.as_view()).unwrap();
    let b = M::param_value("b_scale", &params.as_view()).unwrap();
    let lambda = M::param_value("lambda", &params.as_view()).unwrap();
    let alpha_signed = M::param_value("alpha", &params.as_view()).unwrap();
    let tau = M::param_value("tau", &params.as_view()).unwrap();

    let (alpha, sign) = match alpha_signed
        .partial_cmp(&T::zero())
        .expect("alpha value is NaN")
    {
        Ordering::Less => (-alpha_signed, T::neg(T::one())),
        _ => (alpha_signed, T::one()),
    };

    let (mu, nu, z) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(ObserVec::default()),
            _ => {
                let b_linearized = b / (T::one() - y_offset.powi(2));
                let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

                if mu == T::zero() {
                    let b_q = Vector3::new(T::zero(), T::zero(), b_linearized);
                    let b_s = M::contravariant_vector(
                        &q.as_view(),
                        &b_q.as_view(),
                        &params.as_view(),
                        cs_state,
                    )
                    .expect("failed to construct contravariant basis");

                    Ok(ObserVec::<T, 3>::from(b_s))
                } else {
                    // Bessel function evaluation uses 11 terms.
                    let chi_lff = b_linearized
                        * radius_linearized
                        * sign
                        * bessel_jn(alpha * mu * radius_linearized, 1)
                        / M::detg(
                            &Vector3::new(mu, nu, z).as_view(),
                            &params.as_view(),
                            cs_state,
                        )
                        .unwrap();
                    let xi_lff: T = T::two_pi()
                        * mu
                        * radius_linearized.powi(2)
                        * b_linearized
                        * bessel_jn(alpha * mu * radius_linearized, 0)
                        / M::detg(
                            &Vector3::new(mu, nu, z).as_view(),
                            &params.as_view(),
                            cs_state,
                        )
                        .unwrap();

                    // UT terms.
                    let chi_ut = mu * radius_linearized.powi(2) * b_linearized * tau
                        / (T::one() + (tau * mu * radius_linearized).powi(2))
                        / M::detg(
                            &Vector3::new(mu, nu, z).as_view(),
                            &params.as_view(),
                            cs_state,
                        )
                        .unwrap();
                    let xi_ut = T::two_pi() * mu * radius_linearized.powi(2) * b_linearized
                        / (T::one() + (tau * mu * radius_linearized).powi(2))
                        / M::detg(
                            &Vector3::new(mu, nu, z).as_view(),
                            &params.as_view(),
                            cs_state,
                        )
                        .unwrap();

                    let b_q = Vector3::new(
                        T::zero(),
                        chi_lff * lambda + (T::one() - lambda) * chi_ut,
                        xi_lff * lambda + (T::one() - lambda) * xi_ut,
                    );
                    let b_s = M::contravariant_vector(
                        &q.as_view(),
                        &b_q.as_view(),
                        &params.as_view(),
                        cs_state,
                    )
                    .expect("failed to construct contravariant basis");

                    Ok(ObserVec::<T, 3>::from(b_s))
                }
            }
        },
        None => Ok(ObserVec::default()),
    }
}

macro_rules! impl_cylm {
    ($model: ident, $coords: ident, $docs: literal, $params: expr,
        $(($fn_obs_name: expr, $fn_obs_doc: literal, $fn_obs: expr, $fn_obs_type: ident $(<$( $fn_obs_type_generic:tt ),+>)?)),*)
    => {
        #[doc=$docs]
        #[derive(Debug)]
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

            $(
                paste! {
                    #[doc=$fn_obs_doc]
                    pub fn [<observe_ $fn_obs_name>](
                        scobs: &ScObs<T>,
                        params: &OVector<T, Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>>,
                        _fm_state: &(),
                        cs_state: &XCState<T>) -> Result<$fn_obs_type $(<$( $fn_obs_type_generic ),+>)?, OcnusModelError<T>>

                    {
                        let sc_pos = Vector3::from(match scobs.configuration() {
                            ScObsConf::Position(r) => *r,
                        });

                        let q = match Self::transform_ecs_to_ics(&sc_pos.as_view(), &params.generic_view((0,0), (Const::<{ $coords::<f64>::PARAMS_LEN + $params.len() }>, Const::<1>)), cs_state) {
                            Some(value) => value,
                            None => {
                                return Err(OcnusModelError::CoordinateTransform(sc_pos.into_owned()));
                            }
                        };

                        $fn_obs::<T, Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>, Self>(
                            q,
                            params,
                            cs_state,
                        )
                    }
                }
            ),*

        }

        // Re-implement the OcnusCoords trait because we have no inheritance.
        // Here we make use of the fact that the parameters for the coordinates are at the front
        // and we pass on smaller fixed views of each parameter vector.
        impl<T, P> OcnusCoords<T, Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>, XCState<T>>
            for $model<T, P>
        where
            T: Copy + RealField,
            Self: Send + Sync,
        {
            const PARAMS: OVector<
                &'static str,
                Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>,
            > = OVector::from_array_storage(concat_strs!($coords::<f64>::PARAMS, $params));

            fn contravariant_basis<RStride, CStride>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &XCState<T>,
            ) -> Option<[Vector3<T>; 3]>
            where
                RStride: Dim,
                CStride: Dim,
            {
                $coords::contravariant_basis(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS_LEN }>(0),
                    cs_state,
                )
            }

            fn detg<RStride, CStride>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &XCState<T>,
            ) -> Option<T>
            where
                RStride: Dim,
                CStride: Dim,
            {
                $coords::detg(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS_LEN }>(0),
                    cs_state,
                )
            }

            fn initialize_cs<RStride, CStride>(
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &mut XCState<T>,
            ) where
                RStride: Dim,
                CStride: Dim,
            {
                $coords::initialize_cs(
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS_LEN }>(0),
                    cs_state,
                )
            }

            fn transform_ics_to_ecs<RStride, CStride>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &XCState<T>,
            ) -> Option<Vector3<T>>
            where
                RStride: Dim,
                CStride: Dim,
            {
                $coords::transform_ics_to_ecs(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS_LEN }>(0),
                    cs_state,
                )
            }

            fn transform_ecs_to_ics<RStride, CStride>(
                ecs: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &XCState<T>,
            ) -> Option<Vector3<T>>
            where
                RStride: Dim,
                CStride: Dim,
            {
                $coords::transform_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS_LEN }>(0),
                    cs_state,
                )
            }
        }

        impl<T, P>
            OcnusModel<T, Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>, (), XCState<T>>
            for $model<T, P>
        where
            T: Copy + Default + RealField + SampleUniform,
            for<'x> &'x P: Density<T, Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>>,
            StandardNormal: Distribution<T>,
            usize: AsPrimitive<T>,
            Self: OcnusCoords<T, Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>, XCState<T>>,
        {
            const RCS: usize = 128;

            fn forward<RStride, CStride>(
                &self,
                time_step: T,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>,
                    RStride,
                    CStride,
                >,
                _fm_state: &mut (),
                cs_state: &mut XCState<T>,
            ) -> Result<(), OcnusModelError<T>>
            where
                RStride: Dim,
                CStride: Dim,
            {
                // Extract parameters using their identifiers.
                let vel =
                    Self::param_value("velocity", params).unwrap() / T::from_f64(1.496e8).unwrap();

                cs_state.x += vel * time_step as T;

                Ok(())
            }

            fn initialize_states<RStride, CStride>(
                &self,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>,
                    RStride,
                    CStride,
                >,
                _fm_state: &mut (),
                cs_state: &mut XCState<T>,
            ) -> Result<(), OcnusModelError<T>>
            where
                RStride: Dim,
                CStride: Dim,
            {
                Self::initialize_cs(params, cs_state);

                Ok(())
            }

            fn observe_ics_basis<RStride, CStride>(
                &self,
                scobs: &ScObs<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>,
                    RStride,
                    CStride,
                >,
                _fm_state: &(),
                cs_state: &XCState<T>,
            ) -> Result<ObserVec<T, 12>, OcnusModelError<T>>
            where
                RStride: Dim,
                CStride: Dim,
            {
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

            fn param_step_sizes(
                &self,
            ) -> OVector<T, Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>> {
                OVector::<T, Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>>::from_iterator(
                    (&self.0).get_range().iter().map(|range| {
                        (range.max() - range.min())
                            * T::from_usize(128).unwrap()
                            * T::from_f64(5e-7).unwrap()
                    }),
                )
            }

            fn model_prior(
                &self,
            ) -> impl Density<T, Const<{ $coords::<f64>::PARAMS_LEN + $params.len() }>> {
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
    ("mag", "In situ magnetic field observation", cc_lff_chi_xi, ObserVec<T, 3>)
);

impl_cylm!(
    CCUTModel,
    CCGeometry,
    "Circular-cylindrical uniform twist magnetic flux rope model.",
    ["velocity", "b_scale", "tau"],
    ("mag", "In situ magnetic field observation", cc_ut_chi_xi, ObserVec<T, 3>)
);

impl_cylm!(
    ECHModel,
    ECGeometry,
    "Elliptic-cylindrical uniform twist magnetic flux rope model.",
    ["velocity", "b_scale", "lambda", "alpha", "tau"],
    ("mag", "In situ magnetic field observation", ec_hybrid_obs, ObserVec<T, 3>)
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        base::{OcnusEnsbl, ScObsSeries},
        obser::NullNoise,
        stats::{ConstantDensity, MultivariateDensity, UniformDensity},
    };
    use nalgebra::{DMatrix, Dyn, Matrix, SVector, VecStorage};

    #[test]
    fn test_cclff_model() {
        let prior = MultivariateDensity::<_, Const<8>>::new(&[
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

        let sc = ScObsSeries::<f64>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f64 * 3600.0 * 2.0,
                ScObsConf::Position([1.0, 0.0, 0.0]),
            )
        }));

        let mut ensbl = OcnusEnsbl::new(1, range);
        let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

        ensbl.ptpdf.particles_mut().set_column(
            0,
            &SVector::<f64, 8>::from([
                5.0_f64.to_radians(),
                -3.0_f64.to_radians(),
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
                CCLFFModel::<f64, MultivariateDensity<f64, Const<8>>>::observe_mag,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f64>>,
            )
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 19.562870351).abs() < 1e-6);
        assert!((output[(2, 0)][1] - 20.085491851).abs() < 1e-6);
        assert!((output[(4, 0)][2] + 1.7143621590).abs() < 1e-6);
        assert!((ensbl.cs_states[0].z - 0.025129674).abs() < 1e-6);
    }

    // #[test]
    // fn test_cclff_model_twist() {
    //     let prior = UnivariateND::new([
    //         Uniform1D::new((-1.0, 1.0)).unwrap(),
    //         Uniform1D::new((0.5, 1.0)).unwrap(),
    //         Uniform1D::new((0.05, 0.1)).unwrap(),
    //         Uniform1D::new((0.1, 0.5)).unwrap(),
    //         Constant1D::new(1125.0),
    //         Uniform1D::new((5.0, 100.0)).unwrap(),
    //         Uniform1D::new((-2.4, 2.4)).unwrap(),
    //         Uniform1D::new((0.0, 1.0)).unwrap(),
    //     ]);

    //     let model = CCLFFModel::new(prior);

    //     let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator(
    //         (0..2).map(|i| ScObs::new(0.0, ScObsConf::Distance(i as f64), None)),
    //     );

    //     let mut ensbl = OcnusEnsbl {
    //         params_array: Matrix::<f64, Const<8>, Dyn, VecStorage<f64, Const<8>, Dyn>>::zeros(1),
    //         fm_states: vec![(); 1],
    //         cs_states: vec![XCState::<f64>::default(); 1],
    //         weights: vec![1.0; 1],
    //     };

    //     let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

    //     ensbl.params_array.set_column(
    //         0,
    //         &SVector::<f64, 8>::from([
    //             0_f64.to_radians(),
    //             0_f64.to_radians(),
    //             0.0,
    //             1.56,
    //             0.0,
    //             600.0,
    //             20.0,
    //             2.4048255576957,
    //         ]),
    //     );

    //     model
    //         .initialize_states_ensbl(&sc, &mut ensbl)
    //         .expect("initialization failed");
    //     model
    //         .simulate_ensbl(
    //             &sc,
    //             &mut ensbl,
    //             &mut output.as_view_mut(),
    //             None::<&mut NullNoise<f64>>,
    //         )
    //         .expect("simulation failed");

    //     assert!((output[(0, 0)][1] - 20.0).abs() < 1e-6);
    //     assert!(output[(0, 0)][2].abs() < 1e-6);

    //     assert!(output[(1, 0)][1].abs() < 1e-6);
    //     assert!((output[(1, 0)][2] / bessel_jn(2.4048255576957, 1) - 20.0).abs() < 1e-6);
    // }

    // #[test]
    // fn test_ccut_model() {
    //     let prior = UnivariateND::new([
    //         Uniform1D::new((-1.0, 1.0)).unwrap(),
    //         Uniform1D::new((0.5, 1.0)).unwrap(),
    //         Uniform1D::new((0.05, 0.1)).unwrap(),
    //         Uniform1D::new((0.1, 0.5)).unwrap(),
    //         Uniform1D::new((0.0, 1.0)).unwrap(),
    //         Constant1D::new(1125.0),
    //         Uniform1D::new((5.0, 100.0)).unwrap(),
    //         Uniform1D::new((-10.0, 10.0)).unwrap(),
    //     ]);

    //     let model = CCUTModel::new(prior);

    //     let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..8).map(|i| {
    //         ScObs::new(
    //             224640.0 + i as f64 * 3600.0 * 2.0,
    //             ScObsConf::Distance(1.0),
    //             None,
    //         )
    //     }));

    //     let mut ensbl = OcnusEnsbl {
    //         params_array: Matrix::<f64, Const<8>, Dyn, VecStorage<f64, Const<8>, Dyn>>::zeros(1),
    //         fm_states: vec![(); 1],
    //         cs_states: vec![XCState::default(); 1],
    //         weights: vec![1.0; 1],
    //     };

    //     let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

    //     ensbl.params_array.set_column(
    //         0,
    //         &SVector::<f64, 8>::from([
    //             5.0_f64.to_radians(),
    //             -3.0_f64.to_radians(),
    //             0.1,
    //             0.25,
    //             0.0,
    //             600.0,
    //             20.0,
    //             1.0 / 0.25,
    //         ]),
    //     );

    //     model
    //         .initialize_states_ensbl(&sc, &mut ensbl)
    //         .expect("initialization failed");

    //     model
    //         .simulate_ensbl(
    //             &sc,
    //             &mut ensbl,
    //             &mut output.as_view_mut(),
    //             None::<&mut NullNoise<f64>>,
    //         )
    //         .expect("simulation failed");

    //     assert!((output[(0, 0)][1] - 17.744316275).abs() < 1e-6);
    //     assert!((output[(2, 0)][1] - 19.713773158).abs() < 1e-6);
    //     assert!((output[(4, 0)][2] + 2.3477388063).abs() < 1e-6);
    //     assert!((ensbl.cs_states[0].z - 0.025129674).abs() < 1e-6);
    // }

    // #[test]
    // fn test_ccut_model_twist() {
    //     let prior = UnivariateND::new([
    //         Uniform1D::new((-1.0, 1.0)).unwrap(),
    //         Uniform1D::new((0.5, 1.0)).unwrap(),
    //         Uniform1D::new((0.05, 0.1)).unwrap(),
    //         Uniform1D::new((0.1, 0.5)).unwrap(),
    //         Constant1D::new(1125.0),
    //         Uniform1D::new((5.0, 100.0)).unwrap(),
    //         Uniform1D::new((-2.4, 2.4)).unwrap(),
    //         Uniform1D::new((0.0, 1.0)).unwrap(),
    //     ]);

    //     let model = CCUTModel::new(prior);

    //     let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator(
    //         (0..2).map(|i| ScObs::new(0.0, ScObsConf::Distance(i as f64), None)),
    //     );

    //     let mut ensbl = OcnusEnsbl {
    //         params_array: Matrix::<f64, Const<8>, Dyn, VecStorage<f64, Const<8>, Dyn>>::zeros(1),
    //         fm_states: vec![(); 1],
    //         cs_states: vec![XCState::<f64>::default(); 1],
    //         weights: vec![1.0; 1],
    //     };

    //     let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

    //     ensbl.params_array.set_column(
    //         0,
    //         &SVector::<f64, 8>::from([
    //             0_f64.to_radians(),
    //             0_f64.to_radians(),
    //             0.0,
    //             1.56,
    //             0.0,
    //             600.0,
    //             20.0,
    //             1.0,
    //         ]),
    //     );

    //     model
    //         .initialize_states_ensbl(&sc, &mut ensbl)
    //         .expect("initialization failed");
    //     model
    //         .simulate_ensbl(
    //             &sc,
    //             &mut ensbl,
    //             &mut output.as_view_mut(),
    //             None::<&mut NullNoise<f64>>,
    //         )
    //         .expect("simulation failed");

    //     assert!((output[(0, 0)][1] - 20.0).abs() < 1e-6);
    //     assert!(output[(0, 0)][2].abs() < 1e-6);

    //     assert!((output[(1, 0)][1] - 10.0).abs() < 1e-6);
    //     assert!((output[(1, 0)][2] - 10.0).abs() < 1e-6);
    // }

    // #[test]
    // fn test_ech_model() {
    //     let prior = UnivariateND::new([
    //         Uniform1D::new((-1.0, 1.0)).unwrap(),
    //         Uniform1D::new((-1.0, 1.0)).unwrap(),
    //         Uniform1D::new((-1.0, 1.0)).unwrap(),
    //         Uniform1D::new((0.05, 0.1)).unwrap(),
    //         Uniform1D::new((0.1, 1.0)).unwrap(),
    //         Uniform1D::new((0.1, 0.5)).unwrap(),
    //         Uniform1D::new((0.0, 1.0)).unwrap(),
    //         Constant1D::new(1125.0),
    //         Uniform1D::new((5.0, 100.0)).unwrap(),
    //         Uniform1D::new((0.0, 1.0)).unwrap(),
    //         Uniform1D::new((-10.0, 10.0)).unwrap(),
    //         Uniform1D::new((-10.0, 10.0)).unwrap(),
    //     ]);

    //     let model = ECHModel::new(prior);

    //     let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..8).map(|i| {
    //         ScObs::new(
    //             224640.0 + i as f64 * 3600.0 * 2.0,
    //             ScObsConf::Distance(1.0),
    //             None,
    //         )
    //     }));

    //     let mut ensbl = OcnusEnsbl {
    //         params_array: Matrix::<f64, Const<12>, Dyn, VecStorage<f64, Const<12>, Dyn>>::zeros(1),
    //         fm_states: vec![(); 1],
    //         cs_states: vec![XCState::default(); 1],
    //         weights: vec![1.0; 1],
    //     };

    //     let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

    //     // UT

    //     ensbl.params_array.set_column(
    //         0,
    //         &SVector::<f64, 12>::from([
    //             5.0_f64.to_radians(),
    //             -3.0_f64.to_radians(),
    //             0.0_f64.to_radians(),
    //             0.1,
    //             1.0,
    //             0.25,
    //             0.0,
    //             600.0,
    //             20.0,
    //             0.0,
    //             1.0 / 0.25,
    //             1.0 / 0.25,
    //         ]),
    //     );

    //     model
    //         .initialize_states_ensbl(&sc, &mut ensbl)
    //         .expect("initialization failed");

    //     model
    //         .simulate_ensbl(
    //             &sc,
    //             &mut ensbl,
    //             &mut output.as_view_mut(),
    //             None::<&mut NullNoise<f64>>,
    //         )
    //         .expect("simulation failed");

    //     assert!((output[(0, 0)][1] - 17.744316275).abs() < 1e-6);
    //     assert!((output[(2, 0)][1] - 19.713773158).abs() < 1e-6);
    //     assert!((output[(4, 0)][2] + 2.3477388063).abs() < 1e-6);
    //     assert!((ensbl.cs_states[0].z - 0.025129674).abs() < 1e-6);

    //     // LFF

    //     ensbl.params_array.set_column(
    //         0,
    //         &SVector::<f64, 12>::from([
    //             5.0_f64.to_radians(),
    //             -3.0_f64.to_radians(),
    //             0.0_f64.to_radians(),
    //             0.1,
    //             1.0,
    //             0.25,
    //             0.0,
    //             600.0,
    //             20.0,
    //             1.0,
    //             1.0 / 0.25,
    //             1.0 / 0.25,
    //         ]),
    //     );

    //     model
    //         .initialize_states_ensbl(&sc, &mut ensbl)
    //         .expect("initialization failed");

    //     model
    //         .simulate_ensbl(
    //             &sc,
    //             &mut ensbl,
    //             &mut output.as_view_mut(),
    //             None::<&mut NullNoise<f64>>,
    //         )
    //         .expect("simulation failed");

    //     assert!((output[(0, 0)][1] - 19.562870351).abs() < 1e-6);
    //     assert!((output[(2, 0)][1] - 20.085491851).abs() < 1e-6);
    //     assert!((output[(4, 0)][2] + 1.7143621590).abs() < 1e-6);
    //     assert!((ensbl.cs_states[0].z - 0.025129674).abs() < 1e-6);
    // }
}
