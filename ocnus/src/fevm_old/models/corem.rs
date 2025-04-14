use crate::{
    fevm::{
        FEVM, FEVMError, FEVMNullState, FisherInformation,
        filters::{ABCParticleFilter, ParticleFilter, SIRParticleFilter},
    },
    geom::{CCGeometry, ECGeometry, OcnusCoords, XCState},
    math::bessel_jn,
    obser::ObserVec,
    obser::{ScObs, ScObsConf, ScObsSeries},
    prodef::OcnusProDeF,
};
use nalgebra::{
    Const, Dim, RealField, SVectorView, Scalar, SimdRealField, U1, Vector3, VectorView, VectorView3,
};
use num_traits::{AsPrimitive, Float, FromPrimitive, float::TotalOrder};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, iter::Sum, marker::PhantomData, ops::AddAssign};

/// Linear force-free magnetic field observable.
pub fn cc_lff_obs<T, const P: usize, M, GS>(
    (r, _phi, _z): (T, T, T),
    params: &SVectorView<T, P>,
    _state: &XCState<T>,
) -> Option<Vector3<T>>
where
    T: 'static + Clone + Float + FromPrimitive + TotalOrder,
    M: OcnusCoords<T, P, GS>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("B", params).unwrap();
    let y_offset = M::param_value("y", params).unwrap();
    let alpha_signed = M::param_value("alpha", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    let (alpha, sign) = match alpha_signed.partial_cmp(&T::zero()) {
        Some(ord) => match ord {
            Ordering::Less => (-alpha_signed, T::neg(T::one())),
            _ => (alpha_signed, T::one()),
        },
        None => {
            return None;
        }
    };

    match r.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_linearized = b / (T::one() - y_offset.powi(2));

                // Bessel function evaluation uses 11 terms.
                let b_s = b_linearized * bessel_jn(alpha * r * radius_linearized, 0);
                let b_phi = b_linearized * sign * bessel_jn(alpha * r * radius_linearized, 1);

                Some(Vector3::new(T::zero(), b_phi, b_s))
            }
        },
        None => None,
    }
}

/// Uniform twist magnetic field observable.
pub fn cc_ut_obs<T, const P: usize, M, GS>(
    (r, _phi, _z): (T, T, T),
    params: &SVectorView<T, P>,
    _state: &XCState<T>,
) -> Option<Vector3<T>>
where
    T: Clone + Float + FromPrimitive + TotalOrder,
    M: OcnusCoords<T, P, GS>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("B", params).unwrap();
    let y_offset = M::param_value("y", params).unwrap();
    let tau = M::param_value("tau", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    match r.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_linearized = b / (T::one() - y_offset.powi(2));

                let b_s = b_linearized / (T::one() + tau.powi(2) * (r * radius_linearized).powi(2));
                let b_phi = r * radius_linearized * b_linearized * tau
                    / (T::one() + tau.powi(2) * (r * radius_linearized).powi(2));

                Some(Vector3::new(T::zero(), b_phi, b_s))
            }
        },
        None => None,
    }
}

/// Magnetic field configuration as is used in Nieves-Chinchilla et al. (2018).
pub fn ec_hybrid_obs<T, const P: usize, M, GS>(
    (mu, nu, _s): (T, T, T),
    params: &SVectorView<T, P>,
    _state: &XCState<T>,
) -> Option<Vector3<T>>
where
    T: 'static + Clone + Float + FromPrimitive + TotalOrder,
    M: OcnusCoords<T, P, GS>,
{
    // Extract parameters using their identifiers.
    let y_offset = M::param_value("y", params).unwrap();
    let psi = M::param_value("psi", params).unwrap();
    let delta = M::param_value("delta", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();
    let b = M::param_value("B", params).unwrap();
    let lambda = M::param_value("lambda", params).unwrap();
    let alpha_signed = M::param_value("alpha", params).unwrap();
    let tau = M::param_value("tau", params).unwrap();

    let (alpha, sign) = match alpha_signed.partial_cmp(&T::zero()) {
        Some(ord) => match ord {
            Ordering::Less => (-alpha_signed, T::neg(T::one())),
            _ => (alpha_signed, T::one()),
        },
        None => {
            return None;
        }
    };

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_linearized = b / (T::one() - y_offset.powi(2));
                let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();
                let omega = T::two_pi() * (nu - psi);

                // We remove one radius_lineralized everywhere as it cancels out.
                // See Eqs. 7-9 in Weiss 2024 et al.
                let sqrtg = T::two_pi() * delta.powi(2) * mu * radius_linearized
                    / (omega.cos().powi(2) + delta.powi(2) * omega.sin().powi(2));

                // LFF terms.
                let b_s_lff = T!(2.0)
                    * T::pi()
                    * mu
                    * radius_linearized
                    * b_linearized
                    * bessel_jn(alpha * mu * radius_linearized, 0)
                    / sqrtg;
                let b_nu_lff =
                    b_linearized * sign * bessel_jn(alpha * mu * radius_linearized, 1) / sqrtg;

                // UT terms.
                let b_s_ut = T::two_pi() * mu * radius_linearized * b_linearized
                    / (T::one() + tau.powi(2) * (mu * radius_linearized).powi(2))
                    / sqrtg;
                let b_nu_ut = mu * radius_linearized * b_linearized * tau
                    / (T::one() + tau.powi(2) * (mu * radius_linearized).powi(2))
                    / sqrtg;

                Some(Vector3::new(
                    T::zero(),
                    lambda * b_nu_lff + (T::one() - lambda) * b_nu_ut,
                    lambda * b_s_lff + (T::one() - lambda) * b_s_ut,
                ))
            }
        },
        None => None,
    }
}

macro_rules! concat_arrays {
    ($a: expr, $b: expr) => {{
        let mut c = [$a[0]; $a.len() + $b.len()];

        let mut i1 = 0;
        let mut i2 = 0;

        while i1 < $a.len() {
            c[i1] = $a[i1];

            i1 += 1;
        }

        while i2 < $b.len() {
            c[$a.len() + i2] = $b[i2];

            i2 += 1;
        }

        c
    }};
}

macro_rules! impl_cylm_fevm {
    ($model: ident, $parent: ident, $params: expr, $fn_obs: tt, $docs: literal) => {
        #[doc=$docs]
        #[derive(Debug)]
        pub struct $model<T, D>(pub D, PhantomData<T>)
        where
            T: Float + PartialOrd + RealField + SimdRealField,
            for<'a> &'a D: OcnusProDeF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>;

        impl<T, D> $model<T, D>
        where
            T: Float + PartialOrd + RealField + SimdRealField,
            for<'a> &'a D: OcnusProDeF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`]")]
            pub fn new(pdf: D) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        // Re-implement the OcnusCoords trait because we have no inheritance.
        impl<T, D> OcnusCoords<T, { $parent::<f64>::PARAMS.len() + $params.len() }, XCState<T>>
            for $model<T, D>
        where
            T: Float + PartialOrd + RealField + SimdRealField,
            for<'a> &'a D: OcnusProDeF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
        {
            const PARAMS: [&'static str; { $parent::<f64>::PARAMS.len() + $params.len() }] =
                concat_arrays!($parent::<f64>::PARAMS, $params);

            fn basis_vectors<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState<T>,
            ) -> [Vector3<T>; 3] {
                $parent::basis_vectors(
                    ics,
                    &params.fixed_rows::<{ $parent::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }

            fn coords_ics_to_ecs<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState<T>,
            ) -> Vector3<T> {
                $parent::coords_ics_to_ecs(
                    ics,
                    &params.fixed_rows::<{ $parent::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }

            fn coords_ecs_to_ics<CStride: Dim>(
                ecs: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState<T>,
            ) -> Vector3<T> {
                $parent::coords_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ $parent::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }

            fn create_ecs_vector<CStride: Dim>(
                ics: &VectorView3<T>,
                vec: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState<T>,
            ) -> Vector3<T> {
                $parent::create_ecs_vector(
                    ics,
                    vec,
                    &params.fixed_rows::<{ $parent::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }

            fn geom_state<CStride: Dim>(
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                geom_state: &mut XCState<T>,
            ) {
                $parent::geom_state(
                    &params.fixed_rows::<{ $parent::<f64>::PARAMS.len() }>(0),
                    geom_state,
                )
            }
        }

        impl<T, D>
            FEVM<T, { $parent::<f64>::PARAMS.len() + $params.len() }, 3, FEVMNullState, XCState<T>>
            for $model<T, D>
        where
            T: Copy + Float + RealField + SampleUniform + Scalar + TotalOrder,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
            Self: OcnusCoords<T, { $parent::<f64>::PARAMS.len() + $params.len() }, XCState<T>>,
        {
            const RCS: usize = 128;

            fn fevm_forward(
                &self,
                time_step: T,
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                >,
                _fevm_state: &mut FEVMNullState,
                geom_state: &mut XCState<T>,
            ) -> Result<(), FEVMError<T>> {
                // Extract parameters using their identifiers.
                let vel = Self::param_value("v", params) / T!(1.496e8).unwrap();
                geom_state.T += time_step;
                geom_state.x += vel * time_step as T;

                Ok(())
            }

            fn fevm_observe(
                &self,
                scobs: &ScObs<T, ObserVec<T, 3>>,
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                >,
                _fevm_state: &FEVMNullState,
                geom_state: &XCState<T>,
            ) -> Result<ObserVec<T, 3>, FEVMError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Distance(x) => [*x, T::zero(), T::zero()],
                    ScObsConf::Position(r) => *r,
                });

                let q = Self::coords_ecs_to_ics(&sc_pos.as_view(), params, geom_state);

                let (mu, nu, s) = (q[0], q[1], q[2]);

                let obs = $fn_obs::<
                    T,
                    { $parent::<f64>::PARAMS.len() + $params.len() },
                    Self,
                    XCState<T>,
                >((mu, nu, s), params, geom_state);

                match obs {
                    Some(b_q) => {
                        let b_s = Self::create_ecs_vector(
                            &q.as_view(),
                            &b_q.as_view(),
                            params,
                            geom_state,
                        );

                        Ok(ObserVec::<T, 3>::from(b_s))
                    }
                    None => Ok(ObserVec::default()),
                }
            }

            fn fevm_observe_diagnostics(
                &self,
                scobs: &ScObs<T, ObserVec<T, 3>>,
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                >,
                _fevm_state: &FEVMNullState,
                geom_state: &XCState<T>,
            ) -> Result<ObserVec<T, 12>, FEVMError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Distance(x) => [*x, T::zero(), T::zero()],
                    ScObsConf::Position(r) => *r,
                });

                let q = Self::coords_ecs_to_ics(&sc_pos.as_view(), params, geom_state);

                let (mu, nu, s) = (q[0], q[1], q[2]);

                let [e1, e2, e3] =
                    Self::basis_vectors(&Vector3::from([mu, nu, s]).as_view(), params, geom_state);

                Ok(ObserVec::<T, 12>::from([
                    mu, nu, s, e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2],
                ]))
            }

            fn fevm_state(
                &self,
                _series: &ScObsSeries<T, ObserVec<T, 3>>,
                _params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                >,
                _fevm_state: &mut FEVMNullState,
                _geom_state: &mut XCState<T>,
            ) -> Result<(), FEVMError<T>> {
                Ok(())
            }

            fn fevm_step_sizes(&self) -> [T; { $parent::<f64>::PARAMS.len() + $params.len() }] {
                (&self.0)
                    .get_valid_range()
                    .iter()
                    .map(|(min, max)| (*max - *min) * T!(128.0).unwrap() * T::epsilon())
                    .collect::<Vec<T>>()
                    .try_into()
                    .unwrap()
            }

            fn model_prior(
                &self,
            ) -> impl OcnusProDeF<T, { $parent::<f64>::PARAMS.len() + $params.len() }> {
                &self.0
            }
        }

        impl<T, D>
            ParticleFilter<
                T,
                { $parent::<f64>::PARAMS.len() + $params.len() },
                3,
                FEVMNullState,
                XCState<T>,
            > for $model<T, D>
        where
            T: AsPrimitive<usize>
                + Default
                + Copy
                + Float
                + RealField
                + TotalOrder
                + SampleUniform
                + Scalar
                + Serialize,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T, D>
            ABCParticleFilter<
                T,
                { $parent::<f64>::PARAMS.len() + $params.len() },
                3,
                FEVMNullState,
                XCState<T>,
            > for $model<T, D>
        where
            T: for<'x> AddAssign<&'x T>
                + AsPrimitive<usize>
                + Copy
                + Default
                + Float
                + RealField
                + TotalOrder
                + SampleUniform
                + Scalar
                + Serialize
                + Sum<T>
                + for<'x> Sum<&'x T>,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T, D>
            SIRParticleFilter<
                T,
                { $parent::<f64>::PARAMS.len() + $params.len() },
                3,
                FEVMNullState,
                XCState<T>,
            > for $model<T, D>
        where
            T: for<'x> AddAssign<&'x T>
                + AsPrimitive<usize>
                + Copy
                + Default
                + Float
                + RealField
                + TotalOrder
                + SampleUniform
                + Scalar
                + Serialize
                + Sum<T>
                + for<'x> Sum<&'x T>,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T, D>
            FisherInformation<
                T,
                { $parent::<f64>::PARAMS.len() + $params.len() },
                3,
                FEVMNullState,
                XCState<T>,
            > for $model<T, D>
        where
            T: for<'x> AddAssign<&'x T>
                + Copy
                + Default
                + for<'x> Deserialize<'x>
                + Float
                + FromPrimitive
                + RealField
                + TotalOrder
                + SampleUniform
                + Serialize
                + Scalar
                + Sum<T>,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
        {
        }
    };
}

impl_core_fevm!(
    COREModel,
    COREGeometry,
    ["v", "B", "alpha"],
    cc_lff_obs,
    "Standard 3DCORE magnetic flux rope model."
);
