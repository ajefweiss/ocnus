use crate::{
    fevm::{
        FEVM, FEVMError, FisherInformation,
        filters::{ABCParticleFilter, ParticleFilter, SIRParticleFilter},
    },
    geom::{CCModel, ECModel, OcnusGeometry, XCState},
    math::bessel_jn,
    obser::ObserVec,
    obser::{ScObs, ScObsConf, ScObsSeries},
    stats::PDF,
};
use nalgebra::{
    Const, Dim, RealField, SVectorView, Scalar, SimdRealField, U1, Vector3, VectorView, VectorView3,
};
use num_traits::{AsPrimitive, Float, FromPrimitive, float::TotalOrder};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, iter::Sum, marker::PhantomData, ops::AddAssign};

/// An empty FEVM state.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct FEVMNullState {}

/// Linear force-free magnetic field observable.
pub fn cc_lff_obs<T, const P: usize, M, GS>(
    (r, _phi, _z): (T, T, T),
    params: &SVectorView<T, P>,
    _state: &XCState<T>,
) -> Option<Vector3<T>>
where
    T: 'static + Clone + Float + FromPrimitive + TotalOrder,
    M: OcnusGeometry<T, P, GS>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("B", params);
    let y_offset = M::param_value("y", params);
    let alpha_signed = M::param_value("alpha", params);
    let radius = M::param_value("radius", params);

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
    M: OcnusGeometry<T, P, GS>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("B", params);
    let y_offset = M::param_value("y", params);
    let tau = M::param_value("tau", params);
    let radius = M::param_value("radius", params);

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
    M: OcnusGeometry<T, P, GS>,
{
    // Extract parameters using their identifiers.
    let y_offset = M::param_value("y", params);
    let psi = M::param_value("psi", params);
    let delta = M::param_value("delta", params);
    let radius = M::param_value("radius", params);
    let b = M::param_value("B", params);
    let lambda = M::param_value("lambda", params);
    let alpha_signed = M::param_value("alpha", params);
    let tau = M::param_value("tau", params);

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
                let omega = T::from_f64(2.0 * std::f64::consts::PI).unwrap() * (nu - psi);

                // We remove one radius_lineralized everywhere as it cancels out.
                // See Eqs. 7-9 in Weiss 2024 et al.
                let sqrtg = T::from_f64(2.0 * std::f64::consts::PI).unwrap()
                    * delta.powi(2)
                    * mu
                    * radius_linearized
                    / (omega.cos().powi(2) + delta.powi(2) * omega.sin().powi(2));

                // LFF terms.
                let b_s_lff = T::from_f64(2.0 * std::f64::consts::PI).unwrap()
                    * mu
                    * radius_linearized
                    * b_linearized
                    * bessel_jn(alpha * mu * radius_linearized, 0)
                    / sqrtg;
                let b_nu_lff =
                    b_linearized * sign * bessel_jn(alpha * mu * radius_linearized, 1) / sqrtg;

                // UT terms.
                let b_s_ut = T::from_f64(2.0 * std::f64::consts::PI).unwrap()
                    * mu
                    * radius_linearized
                    * b_linearized
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
            for<'a> &'a D: PDF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>;

        impl<T, D> $model<T, D>
        where
            T: Float + PartialOrd + RealField + SimdRealField,
            for<'a> &'a D: PDF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`]")]
            pub fn new(pdf: D) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        // Re-implement the OcnusGeometry trait because we have no inheritance.
        impl<T, D> OcnusGeometry<T, { $parent::<f64>::PARAMS.len() + $params.len() }, XCState<T>>
            for $model<T, D>
        where
            T: Float + PartialOrd + RealField + SimdRealField,
            for<'a> &'a D: PDF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
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

            fn coords_ics_to_xyz<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState<T>,
            ) -> Vector3<T> {
                $parent::coords_ics_to_xyz(
                    ics,
                    &params.fixed_rows::<{ $parent::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }

            fn coords_xyz_to_ics<CStride: Dim>(
                xyz: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $parent::<f64>::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState<T>,
            ) -> Vector3<T> {
                $parent::coords_xyz_to_ics(
                    xyz,
                    &params.fixed_rows::<{ $parent::<f64>::PARAMS.len() }>(0),
                    state,
                )
            }

            fn create_xyz_vector<CStride: Dim>(
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
                $parent::create_xyz_vector(
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
            for<'a> &'a D: PDF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
            Self: OcnusGeometry<T, { $parent::<f64>::PARAMS.len() + $params.len() }, XCState<T>>,
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
                let vel = Self::param_value("v", params) / T::from_f64(1.496e8).unwrap();
                geom_state.t += time_step;
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

                let q = Self::coords_xyz_to_ics(&sc_pos.as_view(), params, geom_state);

                let (mu, nu, s) = (q[0], q[1], q[2]);

                let obs = $fn_obs::<
                    T,
                    { $parent::<f64>::PARAMS.len() + $params.len() },
                    Self,
                    XCState<T>,
                >((mu, nu, s), params, geom_state);

                match obs {
                    Some(b_q) => {
                        let b_s = Self::create_xyz_vector(
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

                let q = Self::coords_xyz_to_ics(&sc_pos.as_view(), params, geom_state);

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
                    .valid_range()
                    .iter()
                    .map(|(min, max)| (*max - *min) * T::from_f64(128.0).unwrap() * T::epsilon())
                    .collect::<Vec<T>>()
                    .try_into()
                    .unwrap()
            }

            fn model_prior(&self) -> impl PDF<T, { $parent::<f64>::PARAMS.len() + $params.len() }> {
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
            for<'a> &'a D: PDF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
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
            for<'a> &'a D: PDF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
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
            for<'a> &'a D: PDF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
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
            for<'a> &'a D: PDF<T, { $parent::<f64>::PARAMS.len() + $params.len() }>,
        {
        }
    };
}

impl_cylm_fevm!(
    CCLFFModel,
    CCModel,
    ["v", "B", "alpha"],
    cc_lff_obs,
    "Circular-cylindrical linear force-free magnetic flux rope model."
);

impl_cylm_fevm!(
    CCUTModel,
    CCModel,
    ["v", "B", "tau"],
    cc_ut_obs,
    "Circular-cylindrical uniform twist magnetic flux rope model."
);

impl_cylm_fevm!(
    ECHModel,
    ECModel,
    ["v", "B", "lambda", "alpha", "tau"],
    ec_hybrid_obs,
    "Elliptic-cylindrical uniform twist magnetic flux rope model."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fevm::FEVMData,
        stats::{PDFConstant, PDFUniform, PDFUnivariates},
    };
    use nalgebra::{DMatrix, Dyn, Matrix, SVector, VecStorage};

    #[test]
    fn test_cclffmodel() {
        let prior = PDFUnivariates::new([
            PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.5, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.05, 0.1)).unwrap(),
            PDFUniform::new_uvpdf((0.1, 0.5)).unwrap(),
            PDFConstant::new_uvpdf(1125.0),
            PDFUniform::new_uvpdf((5.0, 100.0)).unwrap(),
            PDFUniform::new_uvpdf((-2.4, 2.4)).unwrap(),
            PDFUniform::new_uvpdf((0.0, 1.0)).unwrap(),
        ]);

        let model = CCLFFModel(prior, PhantomData::<f64>);

        let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f64 * 3600.0 * 2.0,
                ScObsConf::Distance(1.0),
                None,
            )
        }));

        let mut data = FEVMData {
            params: Matrix::<f64, Const<8>, Dyn, VecStorage<f64, Const<8>, Dyn>>::zeros(1),
            fevm_states: vec![FEVMNullState::default(); 1],
            geom_states: vec![XCState::default(); 1],
            weights: vec![1.0; 1],
        };

        let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

        data.params.set_column(
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
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");
        model
            .fevm_simulate(&sc, &mut data, &mut output, None)
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 19.562872).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 20.085493).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 1.7143655).abs() < 1e-4);
        assert!((data.geom_states[0].z - 0.025129674).abs() < 1e-4);
    }

    #[test]
    fn test_ccutmodel() {
        let prior = PDFUnivariates::new([
            PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.5, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.05, 0.1)).unwrap(),
            PDFUniform::new_uvpdf((0.1, 0.5)).unwrap(),
            PDFUniform::new_uvpdf((0.0, 1.0)).unwrap(),
            PDFConstant::new_uvpdf(1125.0),
            PDFUniform::new_uvpdf((5.0, 100.0)).unwrap(),
            PDFUniform::new_uvpdf((-10.0, 10.0)).unwrap(),
        ]);

        let model = CCUTModel(prior, PhantomData::<f64>);

        let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f64 * 3600.0 * 2.0,
                ScObsConf::Distance(1.0),
                None,
            )
        }));

        let mut data = FEVMData {
            params: Matrix::<f64, Const<8>, Dyn, VecStorage<f64, Const<8>, Dyn>>::zeros(1),
            fevm_states: vec![FEVMNullState::default(); 1],
            geom_states: vec![XCState::default(); 1],
            weights: vec![1.0; 1],
        };

        let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

        data.params.set_column(
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
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");

        model
            .fevm_simulate(&sc, &mut data, &mut output, None)
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 17.744318).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 19.713774).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 2.3477454).abs() < 1e-4);
        assert!((data.geom_states[0].z - 0.025129674).abs() < 1e-4);
    }

    #[test]
    fn test_echmodel() {
        let prior = PDFUnivariates::new([
            PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.05, 0.1)).unwrap(),
            PDFUniform::new_uvpdf((0.1, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((0.1, 0.5)).unwrap(),
            PDFUniform::new_uvpdf((0.0, 1.0)).unwrap(),
            PDFConstant::new_uvpdf(1125.0),
            PDFUniform::new_uvpdf((5.0, 100.0)).unwrap(),
            PDFUniform::new_uvpdf((0.0, 1.0)).unwrap(),
            PDFUniform::new_uvpdf((-10.0, 10.0)).unwrap(),
            PDFUniform::new_uvpdf((-10.0, 10.0)).unwrap(),
        ]);

        let model = ECHModel::new(prior);

        let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..8).map(|i| {
            ScObs::new(
                224640.0 + i as f64 * 3600.0 * 2.0,
                ScObsConf::Distance(1.0),
                None,
            )
        }));

        let mut data = FEVMData {
            params: Matrix::<f64, Const<12>, Dyn, VecStorage<f64, Const<12>, Dyn>>::zeros(1),
            fevm_states: vec![FEVMNullState::default(); 1],
            geom_states: vec![XCState::default(); 1],
            weights: vec![1.0; 1],
        };

        let mut output = DMatrix::<ObserVec<f64, 3>>::zeros(sc.len(), 1);

        data.params.set_column(
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
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");

        model
            .fevm_simulate(&sc, &mut data, &mut output, None)
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 17.744318).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 19.713774).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 2.3477454).abs() < 1e-4);
        assert!((data.geom_states[0].z - 0.025129674).abs() < 1e-4);

        data.params.set_column(
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
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");

        model
            .fevm_simulate(&sc, &mut data, &mut output, None)
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 19.562872).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 20.085493).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 1.7143655).abs() < 1e-4);
        assert!((data.geom_states[0].z - 0.025129674).abs() < 1e-4);
    }
}
