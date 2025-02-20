use crate::{
    fevm::{FEVMError, FEVM},
    math::{bessel_jn, quaternion_xyz},
    obser::ObserVec,
    scobs::{ScConf, ScObs},
    stats::PDF,
    OcnusModel, OcnusState,
};
use nalgebra::{SVectorView, Vector3, VectorView3};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Model state for analytical cylindrical models.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct XCModelState {
    /// Offset along the time axis.
    pub t: f32,

    /// Offset along the x-axis.
    pub x: f32,

    /// Offset along the z-axis.
    pub z: f32,
}

impl OcnusState for XCModelState {}

/// The circular-cylindric coordinate basis vectors.
pub fn cc_basis<const P: usize, M, S>(
    (_r, phi, _z): (f32, f32, f32),
    _params: &SVectorView<f32, P>,
    _state: &XCModelState,
) -> [Vector3<f32>; 3]
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    let dphi = Vector3::from([-phi.sin(), 0.0, phi.cos()]);
    let dpsi = Vector3::from([0.0, 1.0, 0.0]);

    [Vector3::zeros(), dphi, dpsi]
}

/// The circular-cylindric coordinate transformation (xyz -> ics)
pub fn cc_ics<const P: usize, M, S>(
    (x, y, z): (f32, f32, f32),
    params: &SVectorView<f32, P>,
    _state: &XCModelState,
) -> Vector3<f32>
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    let radius = M::get_param_value("radius", params);
    let y_offset = M::get_param_value("y", params);

    let radius_linearized = radius * (1.0 - y_offset.abs()).sqrt();

    // Compute polar coordinates (r, phi).
    let r = (x.powi(2) + z.powi(2)).sqrt() / radius_linearized;
    let phi = z.atan2(x);

    Vector3::new(r, phi, y)
}

/// Linear force-free magnetic field observable.
pub fn cc_lff_obs<const P: usize, M, S>(
    (r, _phi, _psi): (f32, f32, f32),
    params: &SVectorView<f32, P>,
    _state: &XCModelState,
) -> Option<Vector3<f32>>
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    // Extract parameters using their identifiers.
    let b = M::get_param_value("B", params);
    let y_offset = M::get_param_value("y", params);
    let alpha_signed = M::get_param_value("alpha", params);

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
                let b_phi = b_linearized * sign * bessel_jn(alpha * r, 1);

                Some(Vector3::new(0.0, b_phi, b_s))
            }
        },
        None => None,
    }
}

/// The circular-cylindric coordinate transformation (ics -> xy)
pub fn cc_xyz<const P: usize, M, S>(
    (r, phi, z): (f32, f32, f32),
    params: &SVectorView<f32, P>,
    _state: &XCModelState,
) -> Vector3<f32>
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    let radius = M::get_param_value("radius", params);
    let y_offset = M::get_param_value("y", params);

    let radius_linearized = radius * (1.0 - y_offset.abs()).sqrt();

    // Compute cartesian coordinates (x, y, z).
    let x = phi.cos() * r * radius_linearized;
    let y = phi.sin() * r * radius_linearized;

    Vector3::new(x, y, z)
}

macro_rules! impl_acylm {
    ($model: ident, $params: expr, $param_ranges:expr, $fn_basis: tt, $fn_ics: tt, $fn_xyz: tt, $fn_obs: tt, $docs: literal) => {
        #[doc=$docs]
        #[allow(non_camel_case_types)]
        pub struct $model<T>(pub T);

        impl<T> OcnusModel<{ $params.len() }, XCModelState> for $model<T>
        where
            for<'a> &'a T: PDF<{ $params.len() }>,
        {
            const PARAMS: [&'static str; { $params.len() }] = $params;
            const PARAM_RANGES: [(f32, f32); { $params.len() }] = $param_ranges;

            fn coords_basis_ics(
                ics: &VectorView3<f32>,
                params: &SVectorView<f32, { $params.len() }>,
                state: &XCModelState,
            ) -> [Vector3<f32>; 3] {
                $fn_basis::<{ $params.len() }, Self, XCModelState>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    state,
                )
            }

            fn coords_from_ics(
                ics: &VectorView3<f32>,
                params: &SVectorView<f32, { $params.len() }>,
                state: &XCModelState,
            ) -> Vector3<f32> {
                let phi = Self::get_param_value("phi", params);
                let theta = Self::get_param_value("theta", params);

                let quaternion = quaternion_xyz(phi, 0.0, theta);

                let xyz = $fn_xyz::<{ $params.len() }, Self, XCModelState>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    state,
                );

                quaternion.transform_vector(&xyz) + Vector3::new(state.x, 0.0, state.z)
            }

            fn coords_from_ics_vec(
                ics: &VectorView3<f32>,
                vec: &VectorView3<f32>,
                params: &SVectorView<f32, { $params.len() }>,
                state: &XCModelState,
            ) -> Vector3<f32> {
                let phi = Self::get_param_value("phi", params);
                let theta = Self::get_param_value("theta", params);

                let quaternion = quaternion_xyz(phi, 0.0, theta);

                let [dr, dphi, dpsi] = Self::coords_basis_ics(ics, params, state);

                let vec = dr * vec[0] + dphi * vec[1] + dpsi * vec[2];

                quaternion.transform_vector(&vec)
            }

            fn coords_into_ics(
                xyz: &VectorView3<f32>,
                params: &SVectorView<f32, { $params.len() }>,
                state: &XCModelState,
            ) -> Vector3<f32> {
                let phi = Self::get_param_value("phi", params);
                let theta = Self::get_param_value("theta", params);

                let quaternion = quaternion_xyz(phi, 0.0, theta);
                let xyz_t = quaternion
                    .conjugate()
                    .transform_vector(&(xyz - Vector3::new(state.x, 0.0, state.z)));

                $fn_ics::<{ $params.len() }, Self, XCModelState>(
                    (xyz_t[0], xyz_t[1], xyz_t[2]),
                    params,
                    state,
                )
            }
        }

        impl<T> FEVM<{ $params.len() }, 3, XCModelState> for $model<T>
        where
            T: Sync,
            for<'a> &'a T: PDF<{ $params.len() }>,
            Self: OcnusModel<{ $params.len() }, XCModelState>,
        {
            const RCS: usize = 128;

            fn fevm_forward(
                &self,
                time_step: f32,
                params: &SVectorView<f32, { $params.len() }>,
                state: &mut XCModelState,
            ) -> Result<(), FEVMError> {
                // Extract parameters using their identifiers.
                let vel = Self::get_param_value("v", params) / 1.496e8;
                state.t += time_step;
                state.x += vel * time_step as f32;

                Ok(())
            }

            fn fevm_observe(
                &self,
                scobs: &ScObs<ObserVec<3>>,
                params: &SVectorView<f32, { $params.len() }>,
                state: &XCModelState,
            ) -> Result<ObserVec<3>, FEVMError> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScConf::Distance(x) => [*x, 0.0, 0.0],
                    ScConf::Position(r) => *r,
                });

                let q = Self::coords_into_ics(&sc_pos.as_view(), params, state);

                let (r, phi, z) = (q[0], q[1], q[2]);

                let obs =
                    $fn_obs::<{ $params.len() }, Self, XCModelState>((r, phi, z), params, state);

                match obs {
                    Some(b_q) => {
                        let b_s =
                            Self::coords_from_ics_vec(&q.as_view(), &b_q.as_view(), params, state);

                        Ok(ObserVec::<3>::from(b_s))
                    }
                    None => Ok(ObserVec::default()),
                }
            }
        }
    };
}

// Implementation of the classical linear force-free cylindrical flux rope model.
impl_acylm!(
    CCLFFModel,
    ["phi", "theta", "y", "radius", "v", "B", "alpha", "x_0"],
    [
        (-std::f32::consts::PI / 2.0, std::f32::consts::PI / 2.0),
        (0.0, 2.0 * std::f32::consts::PI),
        (-1.0, 1.0),
        (0.01, 0.5),
        (100.0, 5000.0),
        (0.5, 1000.0),
        (-2.4, 2.4),
        (0.0, 1.0),
    ],
    cc_basis,
    cc_ics,
    cc_xyz,
    cc_lff_obs,
    "Classical linear force-free circular-cylindric flux rope model."
);
