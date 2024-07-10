use crate::{
    OcnusError,
    coords::{CoordsError, OcnusCoords, TTGeometry, TTState},
    fXX,
    forward::{
        FSMError, FisherInformation, OcnusFSM,
        filters::{ABCParticleFilter, ParticleFilter, SIRParticleFilter},
    },
    math::{T, ln, powf, powi},
    obser::{ObserVec, ScObs, ScObsConf, ScObsSeries},
    prodef::OcnusProDeF,
};
use nalgebra::{Const, Dim, SVectorView, U1, Vector3, VectorView, VectorView3};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, marker::PhantomData};

/// Forward model cs_state type for the CORE models.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[allow(missing_docs)]
pub struct COREState<T> {
    pub time: T,
    pub velocity: T,
    pub magnetic_field: T,
}

/// Magnetic field for the CORE model.
pub fn core_obs<T, const P: usize, M>(
    (mu, nu, s): (T, T, T),
    params: &SVectorView<T, P>,
    fm_state: &COREState<T>,
    cs_state: &TTState<T>,
) -> Option<(T, T)>
where
    T: fXX,
    M: OcnusCoords<T, P, TTState<T>>,
{
    // Extract parameters using their identifiers.
    let tau = M::param_value("tau", params).unwrap();

    let magnetic_field = fm_state.magnetic_field;
    let radius = cs_state.minor_radius;

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                if mu == T::zero() {
                    Some((T::zero(), magnetic_field))
                } else {
                    let chi = mu * radius * magnetic_field * tau
                        / (T::one() + powi!(tau, 2) * powi!(mu * radius, 2))
                        * M::detg(&Vector3::new(mu, T::zero(), s).as_view(), params, cs_state)
                            .unwrap()
                        / M::detg(&Vector3::new(mu, nu, s).as_view(), params, cs_state).unwrap();

                    let xi = if ((nu % T::one()) < T!(0.25)) || ((nu % T::one()) > T!(0.75)) {
                        magnetic_field / (T::one() + powi!(tau, 2) * powi!(mu * radius, 2))
                            * M::detg(
                                &Vector3::new(mu, T::zero(), T!(0.5)).as_view(),
                                params,
                                cs_state,
                            )
                            .unwrap()
                            / M::detg(&Vector3::new(mu, nu, s).as_view(), params, cs_state).unwrap()
                    } else {
                        magnetic_field / (T::one() + powi!(tau, 2) * powi!(mu * radius, 2))
                            * M::detg(
                                &Vector3::new(mu, T!(0.5), T!(0.5)).as_view(),
                                params,
                                cs_state,
                            )
                            .unwrap()
                            / M::detg(&Vector3::new(mu, nu, s).as_view(), params, cs_state).unwrap()
                    };

                    Some((chi, xi))
                }
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
macro_rules! impl_core_forward_model {
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
        impl<T, D> OcnusCoords<T, { $coords::<f64>::PARAMS.len() + $params.len() }, TTState<T>>
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
                cs_state: &TTState<T>,
            ) -> Result<[Vector3<T>; 3], CoordsError> {
                $coords::contravariant_basis(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
                )
            }

            fn detg<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                cs_state: &TTState<T>,
            ) -> Result<T, CoordsError> {
                $coords::detg(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
                )
            }

            fn initialize_cs<CStride: Dim>(
                params: &VectorView<
                    T,
                    Const<{ $coords::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                cs_state: &mut TTState<T>,
            ) -> Result<(), CoordsError> {
                $coords::initialize_cs(
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
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
                cs_state: &TTState<T>,
            ) -> Result<Vector3<T>, CoordsError> {
                $coords::transform_ics_to_ecs(
                    ics,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
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
                cs_state: &TTState<T>,
            ) -> Result<Vector3<T>, CoordsError> {
                $coords::transform_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ $coords::<f64>::PARAMS.len() }>(0),
                    cs_state,
                )
            }
        }

        impl<T, D>
            OcnusFSM<
                T,
                ObserVec<T, 3>,
                { $coords::<f64>::PARAMS.len() + $params.len() },
                COREState<T>,
                TTState<T>,
            > for $model<T, D>
        where
            T: fXX,
            D: Sync,
            StandardNormal: Distribution<T>,
            for<'a> &'a D: OcnusProDeF<T, { $coords::<f64>::PARAMS.len() + $params.len() }>,
            Self: OcnusCoords<T, { $coords::<f64>::PARAMS.len() + $params.len() }, TTState<T>>,
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
                fm_state: &mut COREState<T>,
                cs_state: &mut TTState<T>,
            ) -> Result<(), FSMError<T>> {
                // Extract parameters using their identifiers.
                let distance_0 = Self::param_value("distance_0", params).unwrap() * T!(695510.0);
                let diameter_1au = Self::param_value("diameter_1au", params).unwrap();

                let b_scale = Self::param_value("b_scale", params).unwrap();
                let v_0 = Self::param_value("velocity", params).unwrap();
                let v_sw = Self::param_value("sw_velocity", params).unwrap();
                let gamma = Self::param_value("sw_gamma", params).unwrap() * T!(1e-7);

                fm_state.time += time_step;

                let delta_v = v_0 - v_sw;

                let sign = match delta_v.partial_cmp(&T::zero()).unwrap() {
                    Ordering::Greater => T::one(),
                    _ => T::neg(T::one()),
                };

                let rt = (sign / gamma * ln!(T::one() + sign * gamma * delta_v * fm_state.time)
                    + v_sw * fm_state.time
                    + distance_0)
                    / T!(1.496e8);
                let vt = delta_v / (T::one() + sign * gamma * delta_v * fm_state.time) + v_sw;

                cs_state.minor_radius = diameter_1au * powf!(rt, T!(1.14)) / T!(2.0);
                cs_state.major_radius = (rt - cs_state.minor_radius) / T!(2.0);

                fm_state.magnetic_field =
                    b_scale * powf!(T!(2.0) * cs_state.major_radius, T!(-1.68));
                fm_state.velocity = vt;

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
                fm_state: &mut COREState<T>,
                cs_state: &mut TTState<T>,
            ) -> Result<(), OcnusError<T>> {
                // Extract parameters using their identifiers.
                let b_scale = Self::param_value("b_scale", params).unwrap();
                let v_0 = Self::param_value("velocity", params).unwrap();

                Self::initialize_cs(params, cs_state)?;

                fm_state.time = T::zero();
                fm_state.magnetic_field =
                    b_scale * powf!(T!(2.0) * cs_state.major_radius, T!(-1.68));
                fm_state.velocity = v_0;

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
                fm_state: &COREState<T>,
                cs_state: &TTState<T>,
            ) -> Result<ObserVec<T, 3>, OcnusError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Distance(x) => [*x, T::zero(), T::zero()],
                    ScObsConf::Position(r) => *r,
                });

                let q = Self::transform_ecs_to_ics(&sc_pos.as_view(), params, cs_state)?;

                let (mu, nu, s) = (q[0], q[1], q[2]);

                let opt_chi_xi = $fn_obs::<
                    T,
                    { $coords::<f64>::PARAMS.len() + $params.len() },
                    Self
                >((mu, nu, s), params, fm_state, cs_state);

                match opt_chi_xi {
                    Some((chi, xi)) => {
                        let b_q = Vector3::new(T::zero(), chi, xi);
                        let b_s = Self::contravariant_vector_normalized(
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
                _fm_state: &COREState<T>,
                cs_state: &TTState<T>,
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
                COREState<T>,
                TTState<T>,
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
                COREState<T>,
                TTState<T>,
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
                COREState<T>,
                TTState<T>,
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
                COREState<T>,
                TTState<T>,
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
        forward::FSMEnsbl,
        obser::NoNoise,
        prodef::{Constant1D, Uniform1D, UnivariateND},
    };
    use nalgebra::{DMatrix, Dyn, Matrix, SVector, VecStorage};

    #[test]
    fn test_core_circ_model() {
        let prior = UnivariateND::new([
            Uniform1D::new((-1.0, 1.0)).unwrap(),
            Uniform1D::new((0.5, 1.0)).unwrap(),
            Uniform1D::new((-0.5, 0.5)).unwrap(),
            Constant1D::new(20.0),
            Uniform1D::new((0.05, 0.25)).unwrap(),
            Constant1D::new(1.0),
            Constant1D::new(1125.0),
            Uniform1D::new((5.0, 100.0)).unwrap(),
            Uniform1D::new((-10.0, 10.0)).unwrap(),
            Constant1D::new(400.0),
            Constant1D::new(1.0),
        ]);

        let model = COREModel::new(prior);

        let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..10).map(|i| {
            ScObs::new(
                72.0 * 3600.0 + i as f64 * 2.0 * 3600.0,
                ScObsConf::Distance(1.0),
                None,
            )
        }));

        let mut ensbl = FSMEnsbl {
            params_array: Matrix::<f64, Const<11>, Dyn, VecStorage<f64, Const<11>, Dyn>>::zeros(1),
            fm_states: vec![COREState::default(); 1],
            cs_states: vec![TTState::default(); 1],
            weights: vec![1.0; 1],
        };

        let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);
        let mut output_diag = DMatrix::<ObserVec<f64, 12>>::zeros(sc.len(), 1);

        ensbl.params_array.set_column(
            0,
            &SVector::<f64, 11>::from([
                0.0_f64.to_radians(),
                1.0_f64.to_radians(),
                0.0_f64.to_radians(),
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

        model
            .fsm_initialize_states_ensbl(&sc, &mut ensbl)
            .expect("initialization failed");

        model
            .fsm_simulate_ics_plus_basis_ensbl(&sc, &mut ensbl, &mut output_diag.as_view_mut())
            .expect("simulation failed");

        println!("{}", output);
        println!("{}", output_diag);

        // assert!((output[(0, 0)][1] - 17.744318).abs() < 1e-4);
        // assert!((output[(2, 0)][1] - 19.713774).abs() < 1e-4);
        // assert!((output[(4, 0)][2] + 2.3477454).abs() < 1e-4);
    }
}
