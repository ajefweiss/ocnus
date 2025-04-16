use crate::{
    OcnusError,
    coords::{CCGeometry, CoordsError, ECGeometry, OcnusCoords, XCState},
    fXX,
    forward::{
        FSMError, FisherInformation, OcnusFSM,
        filters::{ABCParticleFilter, ParticleFilter, SIRParticleFilter},
    },
    math::{T, bessel_jn, cos, powi, sin, sqrt},
    obser::{ObserVec, ScObs, ScObsConf, ScObsSeries},
    prodef::OcnusProDeF,
};
use nalgebra::{Const, Dim, SVectorView, U1, Vector3, VectorView, VectorView3};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use std::{cmp::Ordering, marker::PhantomData};

/// Linear force-free magnetic field observable.
pub fn cc_lff_obs<T, const P: usize, M, CSST>(
    (r, _phi, _z): (T, T, T),
    params: &SVectorView<T, P>,
    _state: &XCState<T>,
) -> Option<Vector3<T>>
where
    T: fXX,
    M: OcnusCoords<T, P, CSST>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("b_scale", params).unwrap();
    let y_offset = M::param_value("y", params).unwrap();
    let alpha_signed = M::param_value("alpha", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

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
                let b_linearized = b / (T::one() - powi!(y_offset, 2));

                // Bessel function evaluation uses 11 terms.
                let b_s = b_linearized * bessel_jn(alpha * r * radius_linearized, 0);
                let b_phi = b_linearized * sign * bessel_jn(alpha * r * radius_linearized, 1);

                // Account for non-normal basis vectors
                Some(Vector3::new(T::zero(), b_phi / r / radius_linearized, b_s))
            }
        },
        None => None,
    }
}

/// Uniform twist magnetic field observable.
pub fn cc_ut_obs<T, const P: usize, M, CSST>(
    (r, _phi, _z): (T, T, T),
    params: &SVectorView<T, P>,
    _state: &XCState<T>,
) -> Option<Vector3<T>>
where
    T: fXX,
    M: OcnusCoords<T, P, CSST>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("b_scale", params).unwrap();
    let y_offset = M::param_value("y", params).unwrap();
    let tau = M::param_value("tau", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

    match r.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_linearized = b / (T::one() - powi!(y_offset, 2));

                let b_s =
                    b_linearized / (T::one() + powi!(tau, 2) * powi!(r * radius_linearized, 2));
                let b_phi = r * radius_linearized * b_linearized * tau
                    / (T::one() + powi!(tau, 2) * powi!(r * radius_linearized, 2));

                Some(Vector3::new(T::zero(), b_phi / r / radius_linearized, b_s))
            }
        },
        None => None,
    }
}

/// Magnetic field configuration as is used in Nieves-Chinchilla et al. (2018).
pub fn ec_hybrid_obs<T, const P: usize, M, CSST>(
    (mu, nu, _s): (T, T, T),
    params: &SVectorView<T, P>,
    _state: &XCState<T>,
) -> Option<Vector3<T>>
where
    T: fXX,
    M: OcnusCoords<T, P, CSST>,
{
    // Extract parameters using their identifiers.
    let y_offset = M::param_value("y", params).unwrap();
    let psi = M::param_value("psi", params).unwrap();
    let delta = M::param_value("delta", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();
    let b = M::param_value("b_scale", params).unwrap();
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
                let b_linearized = b / (T::one() - powi!(y_offset, 2));
                let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));
                let omega = T::two_pi() * (nu - psi);

                // We remove one radius_linearized everywhere as it cancels out.
                // See Eqs. 7-9 in Weiss 2024 et al.
                let sqrtg = T::two_pi() * powi!(delta, 2) * mu * radius_linearized
                    / (powi!(cos!(omega), 2) + powi!(delta, 2) * powi!(sin!(omega), 2));

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
                    / (T::one() + powi!(tau, 2) * powi!(mu * radius_linearized, 2))
                    / sqrtg;
                let b_nu_ut = mu * radius_linearized * b_linearized * tau
                    / (T::one() + powi!(tau, 2) * powi!(mu * radius_linearized, 2))
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

macro_rules! impl_cylm_forward_model {
    ($model: ident, $coords: ident, $params: expr, $fn_obs: tt, $docs: literal) => {
        #[doc=$docs]
        #[derive(Debug)]
        pub struct $model<T, D>(D, PhantomData<T>)
        where
            T: fXX,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>;

        impl<T, D> $model<T, D>
        where
            T: fXX,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`]")]
            pub fn new(pdf: D) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        // Re-implement the OcnusCoords trait because we have no inheritance.
        // Here we make use of the fact that the parameters for the coordinates are at the front
        // and we pass on smaller fixed views of each parameter vector.
        impl<T, D> OcnusCoords<T, { $coords::<f64>::PARAMS.len() + $params.len() }, XCState<T>>
            for $model<T, D>
        where
            T: fXX,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
            const PARAMS: [&'static str; { $coords::<f64>::PARAMS.len() + $params.len() }] =
                concat_arrays!($coords::<f64>::PARAMS, $params);

            fn contravariant_basis<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState<T>,
            ) -> Result<[Vector3<T>; 3], CoordsError> {
                $coords::contravariant_basis(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }

            fn transform_ics_to_ecs<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState<T>,
            ) -> Result<Vector3<T>, CoordsError> {
                $coords::transform_ics_to_ecs(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }

            fn transform_ecs_to_ics<CStride: Dim>(
                ecs: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState<T>,
            ) -> Result<Vector3<T>, CoordsError> {
                $coords::transform_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }

            fn initialize_cs<CStride: Dim>(
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &mut XCState<T>,
            ) -> Result<(), CoordsError> {
                $coords::initialize_cs(
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }
        }

        impl<T, D>
            OcnusFSM<
                T,
                ObserVec<T, 3>,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                (),
                XCState<T>,
            > for $model<T, D>
        where
            T: fXX,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
            Self: OcnusCoords<T, { $coords::<f64>::PARAMS.len() + $params.len() }, XCState<T>>,
        {
            const RCS: usize = 128;

            fn fsm_forward(
                &self,
                time_step: T,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                >,
                _fm_state: &mut (),
                cs_state: &mut XCState<T>,
            ) -> Result<(), FSMError<T>> {
                // Extract parameters using their identifiers.
                let vel = Self::param_value("velocity", params).unwrap() / T!(1.496e8);

                cs_state.x += vel * time_step as T;

                Ok(())
            }

            fn fsm_initialize_states(
                _series: &ScObsSeries<T, ObserVec<T, 3>>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                >,
                _fm_state: &mut (),
                cs_state: &mut XCState<T>,
            ) -> Result<(), OcnusError<T>> {
                Self::initialize_cs(params, cs_state)?;

                Ok(())
            }

            fn fsm_observe(
                &self,
                scobs: &ScObs<T, ObserVec<T, 3>>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                >,
                _fm_state: &(),
                cs_state: &XCState<T>,
            ) -> Result<ObserVec<T, 3>, OcnusError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Distance(x) => [*x, T::zero(), T::zero()],
                    ScObsConf::Position(r) => *r,
                });

                let q = Self::transform_ecs_to_ics(&sc_pos.as_view(), params, cs_state)?;

                let (mu, nu, s) = (q[0], q[1], q[2]);

                let obs = $fn_obs::<
                    T,
                    { $coords::<f64>::PARAMS.len() + $params.len() },
                    Self,
                    XCState<T>,
                >((mu, nu, s), params, cs_state);

                match obs {
                    Some(b_q) => {
                        let b_s = Self::contravariant_vector(
                            &q.as_view(),
                            &b_q.as_view(),
                            params,
                            cs_state,
                        )?;

                        Ok(ObserVec::<T, 3>::from(b_s))
                    }
                    None => Ok(ObserVec::default()),
                }
            }

            fn fsm_observe_ics_plus_basis(
                &self,
                scobs: &ScObs<T, ObserVec<T, 3>>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                >,
                _fm_state: &(),
                cs_state: &XCState<T>,
            ) -> Result<ObserVec<T, 12>, OcnusError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Distance(x) => [*x, T::zero(), T::zero()],
                    ScObsConf::Position(r) => *r,
                });

                let q = Self::transform_ecs_to_ics(&sc_pos.as_view(), params, cs_state)?;

                let (mu, nu, s) = (q[0], q[1], q[2]);

                let [e1, e2, e3] = Self::contravariant_basis(
                    &Vector3::from([mu, nu, s]).as_view(),
                    params,
                    cs_state,
                )?;

                Ok(ObserVec::<T, 12>::from([
                    mu, nu, s, e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2],
                ]))
            }

            fn param_step_sizes(&self) -> [T; { $coords::<f64>::PARAMS.len() + $params.len() }] {
                (&self.0)
                    .get_valid_range()
                    .iter()
                    .map(|(min, max)| (*max - *min) * T!(128.0) * T::epsilon())
                    .collect::<Vec<T>>()
                    .try_into()
                    .unwrap()
            }

            fn model_prior(
                &self,
            ) -> impl OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }> {
                &self.0
            }
        }

        impl<T, D>
            ParticleFilter<
                T,
                ObserVec<T, 3>,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                (),
                XCState<T>,
            > for $model<T, D>
        where
            T: fXX,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T, D>
            ABCParticleFilter<
                T,
                ObserVec<T, 3>,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                (),
                XCState<T>,
            > for $model<T, D>
        where
            T: fXX + SampleUniform,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T, D>
            SIRParticleFilter<
                T,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                3,
                (),
                XCState<T>,
            > for $model<T, D>
        where
            T: fXX + SampleUniform,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T, D>
            FisherInformation<
                T,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                3,
                (),
                XCState<T>,
            > for $model<T, D>
        where
            T: fXX,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
        {
        }
    };
}

impl_cylm_forward_model!(
    CCLFFModel,
    CCGeometry,
    ["velocity", "b_scale", "alpha"],
    cc_lff_obs,
    "Circular-cylindrical linear force-free magnetic flux rope model."
);

impl_cylm_forward_model!(
    CCUTModel,
    CCGeometry,
    ["velocity", "b_scale", "tau"],
    cc_ut_obs,
    "Circular-cylindrical uniform twist magnetic flux rope model."
);

impl_cylm_forward_model!(
    ECHModel,
    ECGeometry,
    ["velocity", "b_scale", "lambda", "alpha", "tau"],
    ec_hybrid_obs,
    "Elliptic-cylindrical uniform twist magnetic flux rope model."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        forward::FSMEnsbl,
        obser::NoNoise,
        prodef::{Constant1D, Uniform1D, UnivariateND},
    };
    use nalgebra::{DMatrix, Dyn, Matrix, SVector, VecStorage};

    #[test]
    fn test_cclff_model() {
        let prior = UnivariateND::new([
            Uniform1D::new((-1.0, 1.0)).unwrap(),
            Uniform1D::new((0.5, 1.0)).unwrap(),
            Uniform1D::new((0.05, 0.1)).unwrap(),
            Uniform1D::new((0.1, 0.5)).unwrap(),
            Constant1D::new(1125.0),
            Uniform1D::new((5.0, 100.0)).unwrap(),
            Uniform1D::new((-2.4, 2.4)).unwrap(),
            Uniform1D::new((0.0, 1.0)).unwrap(),
        ]);

        let model = CCLFFModel::new(prior);

        let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f64 * 3600.0 * 2.0,
                ScObsConf::Distance(1.0),
                None,
            )
        }));

        let mut ensbl = FSMEnsbl {
            params_array: Matrix::<f64, Const<8>, Dyn, VecStorage<f64, Const<8>, Dyn>>::zeros(1),
            fm_states: vec![(); 1],
            cs_states: vec![XCState::<f64>::default(); 1],
            weights: vec![1.0; 1],
        };

        let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

        ensbl.params_array.set_column(
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
            .fsm_initialize_states_ensbl(&sc, &mut ensbl)
            .expect("initialization failed");
        model
            .fsm_simulate_ensbl(
                &sc,
                &mut ensbl,
                &mut output.as_view_mut(),
                None::<&mut NoNoise<f64>>,
            )
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 19.562872).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 20.085493).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 1.7143655).abs() < 1e-4);
        assert!((ensbl.cs_states[0].z - 0.025129674).abs() < 1e-4);
    }

    #[test]
    fn test_ccut_model() {
        let prior = UnivariateND::new([
            Uniform1D::new((-1.0, 1.0)).unwrap(),
            Uniform1D::new((0.5, 1.0)).unwrap(),
            Uniform1D::new((0.05, 0.1)).unwrap(),
            Uniform1D::new((0.1, 0.5)).unwrap(),
            Uniform1D::new((0.0, 1.0)).unwrap(),
            Constant1D::new(1125.0),
            Uniform1D::new((5.0, 100.0)).unwrap(),
            Uniform1D::new((-10.0, 10.0)).unwrap(),
        ]);

        let model = CCUTModel::new(prior);

        let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f64 * 3600.0 * 2.0,
                ScObsConf::Distance(1.0),
                None,
            )
        }));

        let mut ensbl = FSMEnsbl {
            params_array: Matrix::<f64, Const<8>, Dyn, VecStorage<f64, Const<8>, Dyn>>::zeros(1),
            fm_states: vec![(); 1],
            cs_states: vec![XCState::default(); 1],
            weights: vec![1.0; 1],
        };

        let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

        ensbl.params_array.set_column(
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
            .fsm_initialize_states_ensbl(&sc, &mut ensbl)
            .expect("initialization failed");

        model
            .fsm_simulate_ensbl(
                &sc,
                &mut ensbl,
                &mut output.as_view_mut(),
                None::<&mut NoNoise<f64>>,
            )
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 17.744318).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 19.713774).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 2.3477454).abs() < 1e-4);
        assert!((ensbl.cs_states[0].z - 0.025129674).abs() < 1e-4);
    }

    #[test]
    fn test_ech_model() {
        let prior = UnivariateND::new([
            Uniform1D::new((-1.0, 1.0)).unwrap(),
            Uniform1D::new((-1.0, 1.0)).unwrap(),
            Uniform1D::new((-1.0, 1.0)).unwrap(),
            Uniform1D::new((0.05, 0.1)).unwrap(),
            Uniform1D::new((0.1, 1.0)).unwrap(),
            Uniform1D::new((0.1, 0.5)).unwrap(),
            Uniform1D::new((0.0, 1.0)).unwrap(),
            Constant1D::new(1125.0),
            Uniform1D::new((5.0, 100.0)).unwrap(),
            Uniform1D::new((0.0, 1.0)).unwrap(),
            Uniform1D::new((-10.0, 10.0)).unwrap(),
            Uniform1D::new((-10.0, 10.0)).unwrap(),
        ]);

        let model = ECHModel::new(prior);

        let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f64 * 3600.0 * 2.0,
                ScObsConf::Distance(1.0),
                None,
            )
        }));

        let mut ensbl = FSMEnsbl {
            params_array: Matrix::<f64, Const<12>, Dyn, VecStorage<f64, Const<12>, Dyn>>::zeros(1),
            fm_states: vec![(); 1],
            cs_states: vec![XCState::default(); 1],
            weights: vec![1.0; 1],
        };

        let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

        ensbl.params_array.set_column(
            0,
            &SVector::<f64, 12>::from([
                5.0_f64.to_radians(),
                -3.0_f64.to_radians(),
                0.0_f64.to_radians(),
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
            .fsm_initialize_states_ensbl(&sc, &mut ensbl)
            .expect("initialization failed");

        model
            .fsm_simulate_ensbl(
                &sc,
                &mut ensbl,
                &mut output.as_view_mut(),
                None::<&mut NoNoise<f64>>,
            )
            .expect("simulation failed");

        println!("oo: {}", &output[(0, 0)]);
        println!("oo: {}", &output[(2, 0)]);
        println!("oo: {}", &output[(4, 0)]);

        assert!((output[(0, 0)][1] - 17.744318).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 19.713774).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 2.3477454).abs() < 1e-4);
        assert!((ensbl.cs_states[0].z - 0.025129674).abs() < 1e-4);

        ensbl.params_array.set_column(
            0,
            &SVector::<f64, 12>::from([
                5.0_f64.to_radians(),
                -3.0_f64.to_radians(),
                0.0_f64.to_radians(),
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
            .fsm_initialize_states_ensbl(&sc, &mut ensbl)
            .expect("initialization failed");

        model
            .fsm_simulate_ensbl(
                &sc,
                &mut ensbl,
                &mut output.as_view_mut(),
                None::<&mut NoNoise<f64>>,
            )
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 19.562872).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 20.085493).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 1.7143655).abs() < 1e-4);
        assert!((ensbl.cs_states[0].z - 0.025129674).abs() < 1e-4);
    }
}
