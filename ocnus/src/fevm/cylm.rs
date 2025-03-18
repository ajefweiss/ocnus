use crate::{
    ScObs, ScObsConf, ScObsSeries,
    fevm::{
        FEVM, FEVMError, FisherInformation,
        filters::{ABCParticleFilter, BSParticleFilter, ParticleFilter},
    },
    geometry::OcnusGeometry,
    geometry::{CCModel, XCState},
    math::bessel_jn,
    obser::ObserVec,
    stats::PDF,
};
use nalgebra::{Const, Dim, SVectorView, U1, Vector3, VectorView, VectorView3};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// An empty FEVM state.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct FEVMNullState {}

/// Linear force-free magnetic field observable.
pub fn cc_lff_obs<const P: usize, M, GS>(
    (r, _phi, _psi): (f64, f64, f64),
    params: &SVectorView<f64, P>,
    _state: &XCState,
) -> Option<Vector3<f64>>
where
    M: OcnusGeometry<P, GS>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("B", params);
    let y_offset = M::param_value("y", params);
    let alpha_signed = M::param_value("alpha", params);
    let radius = M::param_value("radius", params);

    let radius_linearized = radius * (1.0 - y_offset.abs().powi(2)).sqrt();

    let (alpha, sign) = match alpha_signed.partial_cmp(&0.0) {
        Some(ord) => match ord {
            Ordering::Less => (-alpha_signed, -1.0),
            _ => (alpha_signed, 1.0),
        },
        None => {
            return None;
        }
    };

    match r.partial_cmp(&1.0) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_linearized = b / (1.0 - y_offset.powi(2));

                // Bessel function evaluation uses 11 terms.
                let b_s = b_linearized * bessel_jn(alpha * r, 0);
                let b_phi = b_linearized * sign * bessel_jn(alpha * r * radius_linearized, 1);

                Some(Vector3::new(0.0, b_phi, b_s))
            }
        },
        None => None,
    }
}

/// Uniform twist magnetic field observable.
pub fn cc_ut_obs<const P: usize, M, GS>(
    (r, _phi, _psi): (f64, f64, f64),
    params: &SVectorView<f64, P>,
    _state: &XCState,
) -> Option<Vector3<f64>>
where
    M: OcnusGeometry<P, GS>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("B", params);
    let y_offset = M::param_value("y", params);
    let tau = M::param_value("tau", params);
    let radius = M::param_value("radius", params);

    let radius_linearized = radius * (1.0 - y_offset.abs().powi(2)).sqrt();

    match r.partial_cmp(&1.0) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_linearized = b / (1.0 - y_offset.powi(2));

                let b_s = b_linearized / (1.0 + tau.powi(2) * (r * radius_linearized).powi(2));
                let b_phi = r * radius_linearized * b_linearized * tau
                    / (1.0 + tau.powi(2) * (r * radius_linearized).powi(2));

                Some(Vector3::new(0.0, b_phi, b_s))
            }
        },
        None => None,
    }
}

/// Magnetic field configuration as is used in Nieves-Chinchilla et al. (2018).
pub fn ec_c10_obs<const P: usize, M, GS>(
    (r, _phi, _psi): (f64, f64, f64),
    params: &SVectorView<f64, P>,
    _state: &XCState,
) -> Option<Vector3<f64>>
where
    M: OcnusGeometry<P, GS>,
{
    // Extract parameters using their identifiers.
    let b = M::param_value("B", params);
    let c_10 = M::param_value("c10", params);
    let delta = M::param_value("delta", params);
    let tau = M::param_value("tau", params);

    match r.partial_cmp(&1.0) {
        Some(ord) => match ord {
            Ordering::Greater => None,
            _ => {
                let b_s = b * delta * (tau - r.powi(2));
                let b_phi = -2.0 * b * delta / (delta.powi(2) + 1.0) / c_10 * r;

                Some(Vector3::new(0.0, b_phi, b_s))
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

macro_rules! impl_fevm {
    ($model: ident, $parent: ident, $params: expr, $param_ranges:expr, $fn_obs: tt, $docs: literal) => {
        #[doc=$docs]
        pub struct $model<T>(pub T)
        where
            for<'a> &'a T: PDF<{ $parent::PARAMS.len() + $params.len() }>;

        impl<T> OcnusGeometry<{ $parent::PARAMS.len() + $params.len() }, XCState> for $model<T>
        where
            for<'a> &'a T: PDF<{ $parent::PARAMS.len() + $params.len() }>,
        {
            const PARAMS: [&'static str; { $parent::PARAMS.len() + $params.len() }] =
                concat_arrays!($parent::PARAMS, $params);
            const PARAM_RANGES: [(f64, f64); { $parent::PARAMS.len() + $params.len() }] =
                concat_arrays!($parent::PARAM_RANGES, $param_ranges);

            fn coords_xyz_vector<CStride: Dim>(
                ics: &VectorView3<f64>,
                vec: &VectorView3<f64>,
                params: &VectorView<
                    f64,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState,
            ) -> Vector3<f64> {
                $parent::coords_xyz_vector(
                    ics,
                    vec,
                    &params.fixed_rows::<{ $parent::PARAMS.len() }>(0),
                    state,
                )
            }

            fn coords_basis<CStride: Dim>(
                ics: &VectorView3<f64>,
                params: &VectorView<
                    f64,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState,
            ) -> [Vector3<f64>; 3] {
                $parent::coords_basis(
                    ics,
                    &params.fixed_rows::<{ $parent::PARAMS.len() }>(0),
                    state,
                )
            }

            fn coords_xyz<CStride: Dim>(
                ics: &VectorView3<f64>,
                params: &VectorView<
                    f64,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState,
            ) -> Vector3<f64> {
                $parent::coords_xyz(
                    ics,
                    &params.fixed_rows::<{ $parent::PARAMS.len() }>(0),
                    state,
                )
            }

            fn coords_ics<CStride: Dim>(
                xyz: &VectorView3<f64>,
                params: &VectorView<
                    f64,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                    U1,
                    CStride,
                >,
                state: &XCState,
            ) -> Vector3<f64> {
                $parent::coords_ics(
                    xyz,
                    &params.fixed_rows::<{ $parent::PARAMS.len() }>(0),
                    state,
                )
            }
        }

        impl<T> FEVM<{ $parent::PARAMS.len() + $params.len() }, 3, FEVMNullState, XCState>
            for $model<T>
        where
            T: Sync,
            for<'a> &'a T: PDF<{ $parent::PARAMS.len() + $params.len() }>,
            Self: OcnusGeometry<{ $parent::PARAMS.len() + $params.len() }, XCState>,
        {
            const RCS: usize = 128;

            fn fevm_forward(
                &self,
                time_step: f64,
                params: &VectorView<
                    f64,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                >,
                _fevm_state: &mut FEVMNullState,
                geom_state: &mut XCState,
            ) -> Result<(), FEVMError> {
                // Extract parameters using their identifiers.
                let vel = Self::param_value("v", params) / 1.496e8;
                geom_state.t += time_step;
                geom_state.x += vel * time_step as f64;

                Ok(())
            }

            fn fevm_observe(
                &self,
                scobs: &ScObs<ObserVec<3>>,
                params: &VectorView<
                    f64,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                >,
                _fevm_state: &FEVMNullState,
                geom_state: &XCState,
            ) -> Result<ObserVec<3>, FEVMError> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Distance(x) => [*x, 0.0, 0.0],
                    ScObsConf::Position(r) => *r,
                });

                let q = Self::coords_ics(&sc_pos.as_view(), params, geom_state);

                let (r, phi, z) = (q[0], q[1], q[2]);

                let obs = $fn_obs::<{ $parent::PARAMS.len() + $params.len() }, Self, XCState>(
                    (r, phi, z),
                    params,
                    geom_state,
                );

                match obs {
                    Some(b_q) => {
                        let b_s = Self::coords_xyz_vector(
                            &q.as_view(),
                            &b_q.as_view(),
                            params,
                            geom_state,
                        );

                        Ok(ObserVec::<3>::from(b_s))
                    }
                    None => Ok(ObserVec::default()),
                }
            }

            fn fevm_state(
                &self,
                _series: &ScObsSeries<ObserVec<3>>,
                params: &VectorView<
                    f64,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                    U1,
                    Const<{ $parent::PARAMS.len() + $params.len() }>,
                >,
                _fevm_state: &mut FEVMNullState,
                geom_state: &mut XCState,
            ) -> Result<(), FEVMError> {
                // Extract parameters using their identifiers.
                let phi = Self::param_value("phi", params);
                let theta = Self::param_value("theta", params);
                let y = Self::param_value("y", params);
                let radius = Self::param_value("radius", params);
                let x_init = Self::param_value("x_0", params);

                geom_state.t = 0.0;
                geom_state.x = x_init;
                geom_state.z =
                    radius * y * ((1.0 - (phi.sin() * theta.cos()).powi(2)) as f64).sqrt()
                        / phi.cos()
                        / theta.cos();

                Ok(())
            }

            fn model_prior(&self) -> impl PDF<{ $parent::PARAMS.len() + $params.len() }> {
                &self.0
            }

            fn validate_model_prior(&self) -> bool {
                (&self.0)
                    .valid_range()
                    .iter()
                    .zip(&Self::PARAM_RANGES)
                    .fold(true, |acc, (pr, mr)| {
                        acc & ((pr.0 >= mr.0) & (pr.1 <= mr.1))
                    })
            }
        }

        impl<T> ParticleFilter<{ $parent::PARAMS.len() + $params.len() }, 3, FEVMNullState, XCState>
            for $model<T>
        where
            T: Sync,
            for<'a> &'a T: PDF<{ $parent::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T>
            ABCParticleFilter<{ $parent::PARAMS.len() + $params.len() }, 3, FEVMNullState, XCState>
            for $model<T>
        where
            T: Sync,
            for<'a> &'a T: PDF<{ $parent::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T>
            BSParticleFilter<{ $parent::PARAMS.len() + $params.len() }, 3, FEVMNullState, XCState>
            for $model<T>
        where
            T: Sync,
            for<'a> &'a T: PDF<{ $parent::PARAMS.len() + $params.len() }>,
        {
        }

        impl<T>
            FisherInformation<{ $parent::PARAMS.len() + $params.len() }, 3, FEVMNullState, XCState>
            for $model<T>
        where
            T: Sync,
            for<'a> &'a T: PDF<{ $parent::PARAMS.len() + $params.len() }>,
        {
        }
    };
}

impl_fevm!(
    CCLFFModel,
    CCModel,
    ["v", "B", "alpha", "x_0"],
    [(250.0, 2500.0), (5.0, 100.0), (-2.4, 2.4), (0.0, 1.0),],
    cc_lff_obs,
    "Circular cylindrical linear force-free magnetic flux rope model."
);

impl_fevm!(
    CCUTModel,
    CCModel,
    ["v", "B", "tau", "x_0"],
    [(250.0, 2500.0), (5.0, 100.0), (-10.0, 10.0), (0.0, 1.0),],
    cc_ut_obs,
    "Circular cylindrical uniform twist magnetic flux rope model."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fevm::{FEVMData, noise::FEVMNoiseNull},
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

        let model = CCLFFModel(prior);

        assert!(model.validate_model_prior());

        let sc = ScObsSeries::<ObserVec<3>>::from_iterator((0..8).map(|i| {
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

        let mut output = DMatrix::<ObserVec<3>>::zeros(sc.len(), 1);

        data.params.set_column(
            0,
            &SVector::<f64, 8>::from([
                5.0_f64.to_radians(),
                -3.0_f64.to_radians(),
                0.025,
                0.2,
                600.0,
                20.0,
                1.0,
                0.0,
            ]),
        );

        model
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");
        model
            .fevm_simulate(&sc, &mut data, &mut output, None::<(&FEVMNoiseNull, u64)>)
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 18.958231).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 19.821177).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 1.874855).abs() < 1e-4);
    }

    #[test]
    fn test_ccutmodel() {
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

        let model = CCUTModel(prior);

        assert!(model.validate_model_prior());

        let sc = ScObsSeries::<ObserVec<3>>::from_iterator((0..8).map(|i| {
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

        let mut output = DMatrix::<ObserVec<3>>::zeros(sc.len(), 1);

        data.params.set_column(
            0,
            &SVector::<f64, 8>::from([
                5.0_f64.to_radians(),
                -3.0_f64.to_radians(),
                0.025,
                0.2,
                600.0,
                20.0,
                1.0,
                0.0,
            ]),
        );

        model
            .fevm_initialize_states_only(&sc, &mut data)
            .expect("initialization failed");
        model
            .fevm_simulate(&sc, &mut data, &mut output, None::<(&FEVMNoiseNull, u64)>)
            .expect("simulation failed");

        assert!((output[(0, 0)][1] - 16.378181).abs() < 1e-4);
        assert!((output[(2, 0)][1] - 19.32089).abs() < 1e-4);
        assert!((output[(4, 0)][2] + 2.687612).abs() < 1e-4);
    }
}
